#!/usr/bin/env python3
"""
finger_analysis_node.py  v2

Changes from v1
---------------
1. Depth sync gate
   - Rejects frame pairs > max_sync_delta_ms apart (default 30 ms).
   - v1 silently used the nearest depth frame regardless of age, so a stalled
     depth stream caused back-projection into seconds-old geometry.

2. Vector EMA replaces per-axis Median
   - Per-axis medians can pick X from frame i, Y from frame j, Z from frame k,
     producing a 3-D point that never physically existed.  Bone lengths would
     then stretch/shrink by several mm per frame even for a stationary hand.
   - Vector EMA keeps all three axes coupled: the filtered point is always a
     geometrically consistent position in 3-D space.
   - Outlier gate: a single depth spike can move the EMA by at most gate_m per
     frame instead of fully capturing it — no window lag penalty.

   Why not alternatives?
     Per-axis Median  : axis-decoupled phantom points (the v1 bug above).
     Vector L1 median : spike-robust but iterative, adds window lag, harder reset.
     1€ per joint     : best adaptive tracking, but 2× state and two params to tune
                        per joint; at 30 Hz with a gate the difference is negligible.
     Kalman           : needs a motion model; finger joints are unpredictable.

3. Adaptive depth patches
   - MCP/PIP: 5×5 (radius 2).  Knuckles are wide; depth is reliable.
   - DIP:     7×7 (radius 3).  Thinner; more holes expected.
   - TIP:     9×9 (radius 4).  Thin fingertip; worst depth quality.
   - Fill-fraction check: joint is rejected if < 25 % of patch pixels are valid,
     preventing a single-pixel unreliable depth value from entering the EMA.

4. Topology enforcement
   - MCP (5) and PIP (6) must both have valid depth; DIP/TIP are optional.
   - v1 called SVD with any number of valid joints — SVD of one point is undefined
     and produces an arbitrary direction that passes silently downstream.

5. Direction vector EMA with sign stabilisation
   - Light EMA (default α = 0.4) on the SVD unit direction eliminates frame-to-frame
     angular jitter without adding lag to the joint position estimates.
   - Before blending, the sign of the new direction is aligned to the previous EMA
     direction to prevent 180° flips from SVD sign ambiguity.

6. Line-fit residual straightness (replaces adjacent-bone cosines)
   Metric: exp(−k · rms_perp / finger_length)
   where rms_perp = RMS perpendicular deviation of all joints from the best-fit line.

   Benefits vs v1:
     • Uses all joints globally — one noisy tip is diluted across the fit, not dominant.
     • No cos < 0 clipping — backward-bent joints are penalised correctly.
     • Normalised by actual finger span — scale-invariant across working distances.
     • Computed on EMA-filtered joints, not raw — far more temporally stable.
     • Threshold semantics (k=10):
         rms_dev 2 % of length → score 0.82  (nearly straight, above 0.65 gate)
         rms_dev 5 % of length → score 0.61  (bent ~30°, below gate)
         rms_dev 10 % of length → score 0.37 (clearly bent)

7. Richer diagnostics
   - Sync rejection counter, topology rejection counter, filter health.
   - Joint topology string ("+-+-") logged every throttle period.
   - Sync delta logged per frame at DEBUG level.
"""

import rospy
import numpy as np
from collections import deque
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from std_msgs.msg import Float32

from hand_detection.msg import Hand
from finger_analysis.msg import FingerAnalysis


# ── Vector EMA ────────────────────────────────────────────────────────────────

class VectorEMA:
    """
    Exponential moving average on a fixed-length numpy vector.

    alpha  — weight on the incoming sample (0 = frozen, 1 = pass-through).
             α = 0.5 at 30 Hz → 90 % step response in ≈ 3 frames (100 ms).
             Median-5 worst-case lag: 5 frames (167 ms).
    gate_m — if the incoming sample is farther than gate_m (metres) from the
             current EMA, alpha is scaled down so the EMA advances by at most
             gate_m per frame.  Eliminates spike capture without window lag.
             0.0 = disabled.
    """

    def __init__(self, alpha: float = 0.5, gate_m: float = 0.05):
        self.alpha  = alpha
        self.gate_m = gate_m
        self._val   = None

    def __call__(self, v: np.ndarray) -> np.ndarray:
        if self._val is None:
            self._val = v.copy()
            return self._val.copy()
        a = self.alpha
        if self.gate_m > 0.0:
            dist = float(np.linalg.norm(v - self._val))
            if dist > self.gate_m:
                a = self.alpha * (self.gate_m / dist)
        self._val = a * v + (1.0 - a) * self._val
        return self._val.copy()

    def reset(self):
        self._val = None

    @property
    def value(self):
        return self._val.copy() if self._val is not None else None


# ── One Euro Filter (scalar) ──────────────────────────────────────────────────

class OneEuroFilter:
    """Casiez 2012 adaptive low-pass filter for scalar signals."""

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x  = None
        self._dx = 0.0
        self._t  = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._t is None:
            self._x = x; self._t = t; return x
        dt = t - self._t
        if dt <= 0:
            return self._x
        a_d  = self._alpha(self.d_cutoff, dt)
        dxh  = a_d * (x - self._x) / dt + (1.0 - a_d) * self._dx
        cut  = self.min_cutoff + self.beta * abs(dxh)
        a    = self._alpha(cut, dt)
        xh   = a * x + (1.0 - a) * self._x
        self._x = xh; self._dx = dxh; self._t = t
        return xh

    def reset(self):
        self._x = None; self._dx = 0.0; self._t = None


# ── Node ──────────────────────────────────────────────────────────────────────

class FingerAnalysisNode:
    # MediaPipe index-finger landmark indices
    MCP = 5   # knuckle       — REQUIRED for topology
    PIP = 6   # proximal IP   — REQUIRED for topology
    DIP = 7   # distal IP     — optional
    TIP = 8   # fingertip     — optional, worst depth quality
    ALL_JOINTS = (MCP, PIP, DIP, TIP)

    # Depth patch half-radius per joint (patch = (2r+1)^2 pixels).
    # Larger for thinner, hole-prone joints.
    PATCH_R = {5: 2, 6: 2, 7: 3, 8: 4}

    # Reject a joint if fewer than this fraction of patch pixels are valid (>0 depth).
    MIN_PATCH_FILL = 0.25

    # Distal-emphasis weights for SVD direction fit.
    # Equal-weight SVD is dominated by MCP+PIP (mandatory, always present) and
    # represents proximal knuckle orientation — not where the fingertip is pointing.
    # Weighting 4:3:2:1 (TIP:DIP:PIP:MCP) pulls the principal axis toward the
    # distal segment, matching human pointing intent.
    JOINT_WEIGHTS = {5: 1.0, 6: 2.0, 7: 3.0, 8: 4.0}

    def __init__(self):
        rospy.init_node('finger_analysis_node', anonymous=False)

        # Camera intrinsics
        self.fx = rospy.get_param('~fx', 901.473)
        self.fy = rospy.get_param('~fy', 899.637)
        self.cx = rospy.get_param('~cx', 642.351)
        self.cy = rospy.get_param('~cy', 349.990)
        self.depth_scale = 0.001  # uint16 mm → m

        # Depth sync gate
        self.max_sync_s = rospy.get_param('~max_sync_delta_ms', 30) / 1000.0
        buf_n           = int(rospy.get_param('~depth_buffer_size', 10))
        self.depth_buf  = deque(maxlen=buf_n)

        # Joint vector EMA
        ema_alpha = float(rospy.get_param('~joint_ema_alpha',  0.5))
        ema_gate  = float(rospy.get_param('~joint_ema_gate_m', 0.05))
        self.joint_ema = {idx: VectorEMA(ema_alpha, ema_gate) for idx in self.ALL_JOINTS}

        # Direction unit-vector EMA (gate disabled — direction is always a unit vector)
        dir_alpha      = float(rospy.get_param('~dir_ema_alpha', 0.4))
        self.dir_ema   = VectorEMA(dir_alpha, gate_m=0.0)
        self._prev_dir = None   # for sign-stabilisation across frames

        # Straightness — One Euro Filter on the line-fit residual score
        self.straight_oef = OneEuroFilter(
            rospy.get_param('~one_euro_min_cutoff', 1.0),
            rospy.get_param('~one_euro_beta',       0.007),
            rospy.get_param('~one_euro_d_cutoff',   1.0))
        self.straight_k = float(rospy.get_param('~straightness_k', 10.0))

        # Hysteresis thresholds
        self.on_thresh  = float(rospy.get_param('~straight_on_threshold',  0.65))
        self.off_thresh = float(rospy.get_param('~straight_off_threshold', 0.45))
        self.is_straight = False

        self._last_raw = None
        self._last_oe  = None

        # Hand-loss filter reset
        self._loss_limit  = int(rospy.get_param('~hand_loss_reset_frames', 5))
        self._frames_lost = 0

        # Diagnostic counters
        self._n_sync_reject = 0
        self._n_topo_reject = 0

        # ROS I/O
        self.bridge     = CvBridge()
        self.finger_pub = rospy.Publisher('/finger/analysis',       FingerAnalysis, queue_size=1)
        self.raw_pub    = rospy.Publisher('/diag/straightness_raw', Float32,        queue_size=1)
        self._jnt_pubs  = {
            5: rospy.Publisher('/diag/knuckle_raw', Point, queue_size=1),
            6: rospy.Publisher('/diag/pip_raw',     Point, queue_size=1),
            7: rospy.Publisher('/diag/dip_raw',     Point, queue_size=1),
            8: rospy.Publisher('/diag/tip_raw',     Point, queue_size=1),
        }

        rospy.Subscriber('/hand/detection',        Hand,  self._hand_cb,  queue_size=1)
        rospy.Subscriber('/d435i/depth/image_raw', Image, self._depth_cb, queue_size=1,
                         buff_size=2**24)

        rospy.loginfo(
            "[FA] v2 ready  sync_gate=%.0fms  joint_EMA(α=%.2f gate=%.0fcm)  "
            "dir_EMA(α=%.2f)  OEF(fc=%.2f β=%.4f)  k=%.1f",
            self.max_sync_s * 1000, ema_alpha, ema_gate * 100, dir_alpha,
            rospy.get_param('~one_euro_min_cutoff', 1.0),
            rospy.get_param('~one_euro_beta', 0.007),
            self.straight_k)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _depth_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
            self.depth_buf.append((msg.header.stamp.to_sec(), img))
        except Exception as e:
            rospy.logerr_throttle(5, "[FA] depth decode: %s", e)

    def _hand_cb(self, msg):
        if not msg.detected:
            self._frames_lost += 1
            if self._frames_lost == self._loss_limit:
                rospy.loginfo("[FA] Hand lost — resetting all filters")
                self._reset()
            return
        self._frames_lost = 0

        depth_img, depth_ts = self._match_depth(msg.header.stamp.to_sec())
        if depth_img is None:
            return
        if len(msg.keypoints_2d) < 9:
            rospy.logwarn_throttle(1.0, "[FA] Only %d keypoints", len(msg.keypoints_2d))
            return

        h, w    = depth_img.shape[:2]
        t_stamp = msg.header.stamp.to_sec()

        # ── Back-project each finger joint to 3D ───────────────────────────
        raw_3d, fill_3d = {}, {}
        for idx in self.ALL_JOINTS:
            kp  = msg.keypoints_2d[idx]
            px  = int(np.clip(kp.x * w, 0, w - 1))
            py  = int(np.clip(kp.y * h, 0, h - 1))
            pt3, fill = self._backproject(px, py, depth_img, self.PATCH_R[idx])
            raw_3d[idx]  = pt3
            fill_3d[idx] = fill
            if pt3 is not None:
                self._jnt_pubs[idx].publish(Point(x=pt3[0], y=pt3[1], z=pt3[2]))

        # ── Topology check: MCP + PIP are mandatory ────────────────────────
        if raw_3d[self.MCP] is None or raw_3d[self.PIP] is None:
            self._n_topo_reject += 1
            rospy.logwarn_throttle(0.5,
                "[FA] MCP/PIP missing  fill=[%.0f%%,%.0f%%,%.0f%%,%.0f%%]  rejects=%d",
                fill_3d[5]*100, fill_3d[6]*100,
                fill_3d.get(7, 0)*100, fill_3d.get(8, 0)*100,
                self._n_topo_reject)
            return

        topo = "".join("+" if raw_3d[i] is not None else "-" for i in self.ALL_JOINTS)
        rospy.logdebug("[FA] joints=%s  sync_delta=%.0fms",
                       topo, abs(depth_ts - t_stamp) * 1000)

        # ── EMA-filter each joint (hold last EMA when raw is unavailable) ──
        filt_3d = {}
        for idx in self.ALL_JOINTS:
            if raw_3d[idx] is not None:
                filt_3d[idx] = self.joint_ema[idx](raw_3d[idx])
            else:
                filt_3d[idx] = self.joint_ema[idx].value  # None if filter not yet seeded

        # ── Direction (SVD on filtered joints → direction EMA) ─────────────
        direction = self._fit_direction(filt_3d)
        if direction is not None:
            direction = self._smooth_dir(direction)

        # ── Straightness (line-fit residual on filtered joints → OEF) ──────
        raw_s = self._straightness(filt_3d)
        if raw_s is not None:
            self.raw_pub.publish(Float32(data=raw_s))
        oe_s = self.straight_oef(raw_s, t_stamp) if raw_s is not None else None
        if raw_s is not None:
            self._last_raw = raw_s
        if oe_s  is not None:
            self._last_oe  = oe_s

        if oe_s is not None:
            if self.is_straight:
                if oe_s < self.off_thresh:
                    self.is_straight = False
            else:
                if oe_s > self.on_thresh:
                    self.is_straight = True

        # If we have no valid direction, suppress the straight flag to prevent
        # the pointing localisation from computing a ray with a zero direction vector.
        if direction is None:
            self.is_straight = False

        # ── Publish ────────────────────────────────────────────────────────
        out = FingerAnalysis()
        out.header.stamp    = msg.header.stamp
        out.header.frame_id = msg.header.frame_id

        kp_mcp = msg.keypoints_2d[self.MCP]
        kp_tip = msg.keypoints_2d[self.TIP]
        out.knuckle_2d.x, out.knuckle_2d.y = kp_mcp.x, kp_mcp.y
        out.tip_2d.x,     out.tip_2d.y     = kp_tip.x, kp_tip.y

        f5 = filt_3d[self.MCP]
        out.knuckle_3d.x = f5[0]
        out.knuckle_3d.y = f5[1]
        out.knuckle_3d.z = f5[2]

        # Populate tip_3d with the best available distal joint (TIP → DIP → PIP).
        # pointing_localization uses this as the ray origin.  Using a distal joint
        # instead of MCP reduces the ray-origin-to-plane distance by ~10-15 cm,
        # cutting angular amplification of direction errors proportionally.
        # z > 0 in the downstream consumer is the validity sentinel (z=0 = not set).
        ray_origin_3d = None
        _ray_label = 'none'
        for _k, _lbl in ((self.TIP, 'TIP'), (self.DIP, 'DIP'), (self.PIP, 'PIP')):
            if filt_3d.get(_k) is not None:
                ray_origin_3d = filt_3d[_k]
                _ray_label = _lbl
                break
        if ray_origin_3d is not None:
            out.tip_3d.x = ray_origin_3d[0]
            out.tip_3d.y = ray_origin_3d[1]
            out.tip_3d.z = ray_origin_3d[2]

        if direction is not None:
            out.direction_3d.x = direction[0]
            out.direction_3d.y = direction[1]
            out.direction_3d.z = direction[2]

        out.straightness_raw      = self._last_raw or 0.0
        out.straightness_one_euro = self._last_oe  or 0.0
        out.straightness_score    = out.straightness_one_euro
        out.is_straight           = self.is_straight
        out.confidence            = (msg.confidences[self.TIP]
                                     if len(msg.confidences) > self.TIP else 0.0)

        self.finger_pub.publish(out)

        rospy.loginfo_throttle(0.5,
            "[FA] joints=%s  str=%.3f→%.3f  straight=%s  origin=%s  sync_delta=%.0fms",
            topo, out.straightness_raw, out.straightness_one_euro,
            self.is_straight, _ray_label, abs(depth_ts - t_stamp) * 1000)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _match_depth(self, target_t: float):
        """
        Return the depth frame closest in time to target_t, or (None, None) if the
        best match exceeds max_sync_s.

        This hard rejection is the key change vs v1.  Stale depth silently corrupted
        3-D reconstructions whenever the depth stream briefly paused.
        """
        if not self.depth_buf:
            rospy.logwarn_throttle(1.0, "[FA] Depth buffer empty")
            return None, None
        best_img, best_ts, best_d = None, None, float('inf')
        for ts, img in self.depth_buf:
            d = abs(ts - target_t)
            if d < best_d:
                best_d, best_img, best_ts = d, img, ts
        if best_d > self.max_sync_s:
            self._n_sync_reject += 1
            rospy.logwarn_throttle(0.5,
                "[FA] Sync rejected: delta=%.0fms > gate=%.0fms  [total: %d]",
                best_d * 1000, self.max_sync_s * 1000, self._n_sync_reject)
            return None, None
        return best_img, best_ts

    def _backproject(self, px: int, py: int, depth_img: np.ndarray, r: int):
        """
        Back-project pixel (px, py) to a 3-D point using median depth of the
        (2r+1)^2 neighbourhood.

        Returns (point_3d, fill_fraction).
        Returns (None, fill) if fewer than MIN_PATCH_FILL of patch pixels are valid.

        Adaptive patch size (r=2/3/4) is the key change vs v1 (fixed 3×3 for all):
        fingertip depth has more holes; a 9×9 patch finds valid pixels even when the
        central pixel is invalid, without reaching adjacent fingers.
        """
        h, w = depth_img.shape[:2]
        y0, y1 = max(0, py - r), min(h, py + r + 1)
        x0, x1 = max(0, px - r), min(w, px + r + 1)
        patch   = depth_img[y0:y1, x0:x1]
        valid   = patch[patch > 0]
        fill    = len(valid) / max(patch.size, 1)

        if fill < self.MIN_PATCH_FILL:
            return None, fill

        z_mm = int(np.median(valid))
        if z_mm == 0 or z_mm > 10000:
            return None, fill

        z = z_mm * self.depth_scale
        return np.array([(px - self.cx) * z / self.fx,
                         (py - self.cy) * z / self.fy,
                         z]), fill

    def _fit_direction(self, filt_3d: dict):
        """
        Distal-weighted SVD line fit through available filtered joints.
        Returns a normalised direction vector pointing MCP → TIP, or None.

        Weighting: TIP=4, DIP=3, PIP=2, MCP=1.
        Equal-weight SVD is dominated by the mandatory proximal joints (MCP/PIP)
        and reflects knuckle orientation, not fingertip orientation.  The 4:1
        weight ratio pulls the principal axis toward the distal segment where
        pointing intent actually lives, reducing the systematic "near me" bias.

        Weighted SVD: scale each centred point by sqrt(w) before decomposition —
        this is algebraically equivalent to minimising sum(w_i * perp_i^2).
        """
        pts, ws = [], []
        for idx in self.ALL_JOINTS:
            if filt_3d[idx] is not None:
                pts.append(filt_3d[idx])
                ws.append(self.JOINT_WEIGHTS[idx])
        if len(pts) < 2:
            return None

        pts_arr  = np.array(pts)
        ws_arr   = np.array(ws)
        centroid = np.average(pts_arr, axis=0, weights=ws_arr)
        centered = pts_arr - centroid
        _, _, Vt = np.linalg.svd(centered * np.sqrt(ws_arr[:, None]), full_matrices=False)
        d = Vt[0]

        # Sign: must point from MCP toward the most distal available joint
        distal = None
        for k in (self.TIP, self.DIP, self.PIP):
            if filt_3d.get(k) is not None:
                distal = filt_3d[k]
                break
        if distal is not None and np.dot(d, distal - filt_3d[self.MCP]) < 0:
            d = -d

        n = float(np.linalg.norm(d))
        return d / n if n > 1e-6 else None

    def _smooth_dir(self, d: np.ndarray) -> np.ndarray:
        """
        EMA on the direction unit vector.

        Sign is aligned to the previous EMA direction before blending to prevent
        180° jumps from SVD sign ambiguity (Vt[0] and -Vt[0] are both valid SVD
        outputs for the same line; adjacent frames can disagree on sign).
        """
        if self._prev_dir is not None and np.dot(d, self._prev_dir) < 0:
            d = -d
        smoothed = self.dir_ema(d)
        n = float(np.linalg.norm(smoothed))
        result = smoothed / n if n > 1e-6 else d
        self._prev_dir = result
        return result

    def _straightness(self, filt_3d: dict):
        """
        Straightness from SVD line-fit residual.

        score = exp(−k · rms_perp / finger_length)

        rms_perp = RMS perpendicular distance of each joint from the best-fit line.
        finger_length = max pairwise distance across available joints.

        Requires ≥ 3 joints to form a meaningful bending measure (2 joints always
        lie on a line; you cannot detect curvature with fewer than 3 points).
        Holds the last OEF value if fewer joints are available.
        """
        pts = [filt_3d[i] for i in self.ALL_JOINTS if filt_3d[i] is not None]
        if len(pts) < 3:
            return None

        arr      = np.array(pts)
        centroid = arr.mean(axis=0)
        centered = arr - centroid

        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        axis = Vt[0]

        # Finger length: use max pairwise distance so it reflects actual span
        finger_len = max(
            float(np.linalg.norm(arr[i] - arr[j]))
            for i in range(len(arr)) for j in range(i + 1, len(arr)))
        if finger_len < 1e-4:
            return None

        # RMS perpendicular deviation from the best-fit line
        projs   = centered @ axis                       # scalar projections (n,)
        on_line = np.outer(projs, axis)                 # (n, 3) points on line
        perp    = centered - on_line                    # (n, 3) perpendicular vectors
        rms_dev = float(np.sqrt(np.mean(np.sum(perp**2, axis=1))))

        return float(np.exp(-self.straight_k * rms_dev / finger_len))

    def _reset(self):
        for ema in self.joint_ema.values():
            ema.reset()
        self.dir_ema.reset()
        self.straight_oef.reset()
        self._prev_dir   = None
        self._last_raw   = None
        self._last_oe    = None
        self.is_straight = False


if __name__ == '__main__':
    try:
        node = FingerAnalysisNode()
        rospy.spin()
    except KeyboardInterrupt:
        pass
