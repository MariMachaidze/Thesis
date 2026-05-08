#!/usr/bin/env python3
"""
Filter Live Tuner — four panels, two filters.

Run alongside the full system:
    Terminal 1: roslaunch visualization thesis_system.launch
    Terminal 2: python3 thesis_ws/oef_tuner.py

Panel 1 — Straightness score   [OEF]    sliders: min_cutoff, beta
Panel 2 — All 4 finger joints Z [Median] slider:  window (integer)
           MCP=light-gray  PIP=tan  DIP=brown  TIP=charcoal
           Blue = system Median-5 on MCP   Red = candidate Median-N on MCP
Panel 3 — Pointing U (left←→right on paper)  [OEF]
Panel 4 — Pointing V (top↕bottom on paper)   [OEF]
           Right y-axis of panels 3&4 shows cm (paper is 85×60 cm).
           U=0 is left edge, U=1 is right edge.
           V=0 is top edge,  V=1 is bottom edge.
           Gray=raw  Blue=system  Red=candidate  Dashed=out-of-[0,1] region

'Print Params' prints launch-file XML.
'Start Recording' logs all traces to oef_tune_<timestamp>.csv.
"""

import rospy
import numpy as np
import threading
import csv
import time as _time
from collections import deque

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec

from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from finger_analysis.msg import FingerAnalysis
from pointing_localization.msg import PointingTarget


WINDOW_S  = 15
UPDATE_MS = 120
MAX_PTS   = 15 * 35

PAPER_X_CM = 85.0
PAPER_Y_CM = 60.0


# ---------------------------------------------------------------------------
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._reset()

    def _reset(self):
        self.x_prev = None; self.dx_prev = 0.0; self.t_prev = None

    def _alpha(self, cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.t_prev = x, t; return x
        dt = t - self.t_prev
        if dt <= 0: return self.x_prev
        a_d    = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * (x - self.x_prev) / dt + (1 - a_d) * self.dx_prev
        a      = self._alpha(self.min_cutoff + self.beta * abs(dx_hat), dt)
        x_hat  = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat; self.dx_prev = dx_hat; self.t_prev = t
        return x_hat

    def reconfigure(self, min_cutoff, beta):
        self.min_cutoff = min_cutoff; self.beta = beta; self._reset()


class MedianFilter:
    def __init__(self, window=5):
        self.buf = deque(maxlen=int(window))

    def __call__(self, x):
        self.buf.append(x); return float(np.median(self.buf))

    def reconfigure(self, window):
        self.buf = deque(maxlen=int(window))

    def reset(self):
        self.buf.clear()


# ---------------------------------------------------------------------------
class OEFTuner:

    def __init__(self):
        rospy.init_node('oef_tuner', anonymous=True)
        self._lock = threading.Lock()
        self._recording = False; self._csv_writer = None; self._csv_file = None

        # knuckle: (t, raw_z, sys_z, cand_z)
        self.mcp_buf = deque(maxlen=MAX_PTS)
        # other joints: (t, raw_z) only — no median output published for these
        self.pip_buf = deque(maxlen=MAX_PTS)
        self.dip_buf = deque(maxlen=MAX_PTS)
        self.tip_buf = deque(maxlen=MAX_PTS)
        # uv: (t, raw_u, sys_u, cand_u, raw_v, sys_v, cand_v)
        self.uv_buf  = deque(maxlen=MAX_PTS)
        # straightness: (t, raw, sys, cand)
        self.str_buf = deque(maxlen=MAX_PTS)

        self.str_filt  = OneEuroFilter(1.0, 0.007, 1.0)
        self.jnt_filt  = MedianFilter(5)
        self.uv_filt_u = OneEuroFilter(1.0, 0.8, 1.0)
        self.uv_filt_v = OneEuroFilter(1.0, 0.8, 1.0)

        self._str_sys  = None
        self._jnt_sys  = None   # knuckle_3d.z from /finger/analysis
        self._uv_sys_u = None
        self._uv_sys_v = None

        # Current slider values — updated by callbacks, written to every CSV row
        self._params = dict(str_mc=1.0, str_beta=0.007, jnt_win=5,
                            uv_mc=1.0,  uv_beta=0.8)

        rospy.Subscriber('/diag/straightness_raw', Float32,        self._str_cb)
        rospy.Subscriber('/diag/knuckle_raw',      Point,          self._mcp_cb)
        rospy.Subscriber('/diag/pip_raw',           Point,          self._pip_cb)
        rospy.Subscriber('/diag/dip_raw',           Point,          self._dip_cb)
        rospy.Subscriber('/diag/tip_raw',           Point,          self._tip_cb)
        rospy.Subscriber('/diag/pointing_raw',      Point,          self._pt_raw_cb)
        rospy.Subscriber('/finger/analysis',        FingerAnalysis, self._fa_cb)
        rospy.Subscriber('/pointing/target',        PointingTarget, self._pt_filt_cb)

        rospy.loginfo("Filter Tuner ready")

    # ------------------------------------------------------------------ callbacks

    def _fa_cb(self, msg):
        with self._lock:
            self._str_sys = msg.straightness_one_euro
            self._jnt_sys = msg.knuckle_3d.z

    def _pt_filt_cb(self, msg):
        with self._lock:
            if msg.is_valid:
                self._uv_sys_u = msg.u_normalized
                self._uv_sys_v = msg.v_normalized

    def _str_cb(self, msg):
        t = rospy.Time.now().to_sec(); raw = float(msg.data)
        cand = self.str_filt(raw, t)
        with self._lock:
            sys = self._str_sys if self._str_sys is not None else raw
            self.str_buf.append((t, raw, sys, cand))
            self._record('straight', t, raw, sys, cand)

    def _mcp_cb(self, msg):
        t = rospy.Time.now().to_sec(); raw = float(msg.z)
        cand = self.jnt_filt(raw)
        with self._lock:
            sys = self._jnt_sys if self._jnt_sys is not None else raw
            self.mcp_buf.append((t, raw, sys, cand))
            self._record('mcp_z', t, raw, sys, cand)

    def _pip_cb(self, msg):
        t = rospy.Time.now().to_sec()
        with self._lock:
            self.pip_buf.append((t, float(msg.z)))

    def _dip_cb(self, msg):
        t = rospy.Time.now().to_sec()
        with self._lock:
            self.dip_buf.append((t, float(msg.z)))

    def _tip_cb(self, msg):
        t = rospy.Time.now().to_sec()
        with self._lock:
            self.tip_buf.append((t, float(msg.z)))

    def _pt_raw_cb(self, msg):
        t = rospy.Time.now().to_sec()
        raw_u = float(msg.x); raw_v = float(msg.y)
        cand_u = self.uv_filt_u(raw_u, t)
        cand_v = self.uv_filt_v(raw_v, t)
        with self._lock:
            su = self._uv_sys_u if self._uv_sys_u is not None else raw_u
            sv = self._uv_sys_v if self._uv_sys_v is not None else raw_v
            self.uv_buf.append((t, raw_u, su, cand_u, raw_v, sv, cand_v))
            self._record('uv_u', t, raw_u, su, cand_u)
            self._record('uv_v', t, raw_v, sv, cand_v)

    def _record(self, signal, t, raw, sys, cand):
        if self._recording and self._csv_writer:
            p = self._params
            self._csv_writer.writerow([
                signal, f'{t:.6f}', f'{raw:.6f}', f'{sys:.6f}', f'{cand:.6f}',
                f'{p["str_mc"]:.3f}', f'{p["str_beta"]:.3f}', int(p["jnt_win"]),
                f'{p["uv_mc"]:.3f}', f'{p["uv_beta"]:.3f}',
            ])

    # ------------------------------------------------------------------ GUI

    def build_gui(self):
        fig = plt.figure(figsize=(15, 14))
        fig.suptitle(
            "Filter Tuner  |  gray=raw   blue=system   red=candidate\n"
            "Panel 1: Straightness [OEF]   Panel 2: Joint depths [Median]   "
            "Panels 3-4: Pointing UV [OEF]",
            fontsize=10, fontweight='bold')

        gs = GridSpec(nrows=15, ncols=2, figure=fig,
                      hspace=0.75, wspace=0.35,
                      left=0.07, right=0.93, top=0.93, bottom=0.04)

        # ---- axes ----
        ax_str = fig.add_subplot(gs[0:3,  :])
        ax_jnt = fig.add_subplot(gs[4:7,  :])
        ax_u   = fig.add_subplot(gs[8:10, :])
        ax_v   = fig.add_subplot(gs[11:13, :])

        ax_str.set_title("Straightness score  [OEF]   on=0.65  off=0.45", fontsize=9)
        ax_jnt.set_title(
            "All finger joints — Z depth (m)   "
            "MCP=light-gray  PIP=tan  DIP=brown  TIP=charcoal   "
            "blue=system Median-5(MCP)  red=candidate",
            fontsize=9)
        ax_u.set_title(
            "Pointing U — horizontal position on paper  [OEF]\n"
            "0 = left edge   1 = right edge   right axis = cm", fontsize=9)
        ax_v.set_title(
            "Pointing V — vertical position on paper  [OEF]\n"
            "0 = top edge   1 = bottom edge   right axis = cm", fontsize=9)

        for ax in (ax_str, ax_jnt, ax_u, ax_v):
            ax.set_xlabel("time (s)", fontsize=8)
            ax.tick_params(labelsize=8)

        ax_str.axhline(0.65, color='lime',   lw=1, ls='--', alpha=0.8, label='on=0.65')
        ax_str.axhline(0.45, color='orange', lw=1, ls='--', alpha=0.8, label='off=0.45')
        ax_str.set_ylim(-0.05, 1.1)

        for ax, paper_cm, label_lo, label_hi in [
                (ax_u, PAPER_X_CM, 'left edge', 'right edge'),
                (ax_v, PAPER_Y_CM, 'top edge',  'bottom edge')]:
            ax.axhline(0.0, color='gray', lw=0.8, ls='-')
            ax.axhline(1.0, color='gray', lw=0.8, ls='-')
            ax.axhspan(-0.5, 0.0, color='red', alpha=0.06)
            ax.axhspan(1.0,  1.5, color='red', alpha=0.06)
            ax.set_ylim(-0.15, 1.15)
            ax.text(0, -0.12, f'← {label_lo}', fontsize=7, color='gray',
                    transform=ax.get_yaxis_transform())
            ax.text(0,  1.03, f'← {label_hi}', fontsize=7, color='gray',
                    transform=ax.get_yaxis_transform())
            ax2 = ax.twinx()
            ax2.set_ylim(-0.15 * paper_cm, 1.15 * paper_cm)
            ax2.set_ylabel("cm", fontsize=8)
            ax2.tick_params(labelsize=7)

        GRAY, BLUE, RED = '#aaaaaa', '#4477cc', '#cc3333'
        C_PIP, C_DIP, C_TIP = '#cc9955', '#885522', '#443322'

        # straightness lines
        ln_sr, = ax_str.plot([], [], color=GRAY, lw=1,   alpha=0.7, label='raw')
        ln_ss, = ax_str.plot([], [], color=BLUE, lw=1.5, label='system OEF')
        ln_sc, = ax_str.plot([], [], color=RED,  lw=1.5, label='candidate OEF')
        ax_str.legend(loc='lower right', fontsize=8, ncol=5)

        # joint lines
        ln_mcp_r, = ax_jnt.plot([], [], color=GRAY,  lw=1, alpha=0.6, label='MCP raw')
        ln_pip_r, = ax_jnt.plot([], [], color=C_PIP, lw=1, alpha=0.7, label='PIP raw')
        ln_dip_r, = ax_jnt.plot([], [], color=C_DIP, lw=1, alpha=0.7, label='DIP raw')
        ln_tip_r, = ax_jnt.plot([], [], color=C_TIP, lw=1, alpha=0.7, label='TIP raw')
        ln_mcp_s, = ax_jnt.plot([], [], color=BLUE,  lw=2,             label='MCP system Median-5')
        ln_mcp_c, = ax_jnt.plot([], [], color=RED,   lw=1.5, ls='--',  label='MCP candidate Median-N')
        ax_jnt.legend(loc='upper left', fontsize=7, ncol=3)

        # U lines
        ln_ur, = ax_u.plot([], [], color=GRAY, lw=1,   alpha=0.7, label='raw U')
        ln_us, = ax_u.plot([], [], color=BLUE, lw=1.5,             label='system OEF')
        ln_uc, = ax_u.plot([], [], color=RED,  lw=1.5,             label='candidate OEF')
        ax_u.legend(loc='upper left', fontsize=8)

        # V lines
        ln_vr, = ax_v.plot([], [], color=GRAY, lw=1,   alpha=0.7, label='raw V')
        ln_vs, = ax_v.plot([], [], color=BLUE, lw=1.5,             label='system OEF')
        ln_vc, = ax_v.plot([], [], color=RED,  lw=1.5,             label='candidate OEF')
        ax_v.legend(loc='upper left', fontsize=8)

        # ---- sliders ----
        def add_oef_sliders(gs_row, label, init_mc, init_beta):
            sl_mc   = Slider(fig.add_subplot(gs[gs_row, 0]),
                             f'{label}  min_cutoff', 0.05, 10.0,
                             valinit=init_mc,   valstep=0.05, color='#4477cc')
            sl_beta = Slider(fig.add_subplot(gs[gs_row, 1]),
                             f'{label}  beta',       0.0,  20.0,
                             valinit=init_beta, valstep=0.05, color='#cc4444')
            return sl_mc, sl_beta

        sl_str_mc, sl_str_beta = add_oef_sliders(3,  'Straight', 1.0, 0.007)

        ax_jw = fig.add_subplot(gs[7, :])
        sl_jnt_win = Slider(ax_jw, 'Joint Median  window (frames)', 1, 15,
                            valinit=5, valstep=1, color='#44aa44')

        sl_uv_mc, sl_uv_beta = add_oef_sliders(13, 'UV', 1.0, 0.8)

        # ---- buttons ----
        ax_btn_rec   = fig.add_axes([0.07, 0.002, 0.18, 0.025])
        ax_btn_print = fig.add_axes([0.27, 0.002, 0.18, 0.025])
        btn_rec   = Button(ax_btn_rec,   'Start Recording', color='#eeffee')
        btn_print = Button(ax_btn_print, 'Print Params',    color='#eeeeff')

        # ---- slider callbacks ----
        def on_str(val):
            self.str_filt.reconfigure(sl_str_mc.val, sl_str_beta.val)
            self._params.update(str_mc=sl_str_mc.val, str_beta=sl_str_beta.val)

        def on_jnt(val):
            w = int(sl_jnt_win.val)
            self.jnt_filt.reconfigure(w)
            self._params['jnt_win'] = w
            ln_mcp_c.set_label(f'MCP candidate Median-{w}')
            ax_jnt.legend(loc='upper left', fontsize=7, ncol=3)

        def on_uv(val):
            self.uv_filt_u.reconfigure(sl_uv_mc.val, sl_uv_beta.val)
            self.uv_filt_v.reconfigure(sl_uv_mc.val, sl_uv_beta.val)
            self._params.update(uv_mc=sl_uv_mc.val, uv_beta=sl_uv_beta.val)

        sl_str_mc.on_changed(on_str);  sl_str_beta.on_changed(on_str)
        sl_jnt_win.on_changed(on_jnt)
        sl_uv_mc.on_changed(on_uv);    sl_uv_beta.on_changed(on_uv)

        # ---- record ----
        def toggle_record(event):
            if not self._recording:
                fname = f"oef_tune_{int(_time.time())}.csv"
                self._csv_file   = open(fname, 'w', newline='')
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow([
                    'signal','time','raw','system','candidate',
                    'str_mc','str_beta','jnt_win','uv_mc','uv_beta',
                ])
                self._recording = True
                btn_rec.label.set_text('Stop Recording')
                btn_rec.ax.set_facecolor('#ffeeee')
                rospy.loginfo(f"[Tuner] Recording → {fname}")
            else:
                self._recording = False
                if self._csv_file:
                    self._csv_file.close()
                    self._csv_file = None; self._csv_writer = None
                btn_rec.label.set_text('Start Recording')
                btn_rec.ax.set_facecolor('#eeffee')
                rospy.loginfo("[Tuner] Recording stopped")
            fig.canvas.draw_idle()
        btn_rec.on_clicked(toggle_record)

        # ---- print params ----
        def print_params(event):
            print("\n" + "="*60)
            print("<!-- finger_analysis — paste into thesis_system.launch -->")
            print(f'    <param name="joint_median_window" value="{int(sl_jnt_win.val)}"/>')
            print(f'    <param name="one_euro_min_cutoff" value="{sl_str_mc.val:.3f}"/>')
            print(f'    <param name="one_euro_beta"       value="{sl_str_beta.val:.3f}"/>')
            print(f'    <param name="one_euro_d_cutoff"   value="1.0"/>')
            print("<!-- pointing_localization -->")
            print(f'    <param name="filter_min_cutoff" value="{sl_uv_mc.val:.3f}"/>')
            print(f'    <param name="filter_beta"        value="{sl_uv_beta.val:.3f}"/>')
            print(f'    <param name="filter_d_cutoff"    value="1.0"/>')
            print("="*60 + "\n")
        btn_print.on_clicked(print_params)

        # ---- animation ----
        def _trim(buf):
            if len(buf) < 2: return None
            arr = np.array(buf)
            mask = arr[:, 0] >= arr[-1, 0] - WINDOW_S
            arr = arr[mask]
            if len(arr) < 2: return None
            arr[:, 0] -= arr[0, 0]   # relative time
            return arr

        def update(frame):
            if rospy.is_shutdown(): return
            with self._lock:
                sd = list(self.str_buf)
                md = list(self.mcp_buf)
                pd = list(self.pip_buf)
                dd = list(self.dip_buf)
                td = list(self.tip_buf)
                ud = list(self.uv_buf)

            # --- straightness ---
            a = _trim(sd)
            if a is not None:
                ln_sr.set_data(a[:,0], a[:,1])
                ln_ss.set_data(a[:,0], a[:,2])
                ln_sc.set_data(a[:,0], a[:,3])
                ax_str.set_xlim(max(0, a[-1,0]-WINDOW_S), a[-1,0]+0.3)

            # --- joints Z ---
            a = _trim(md)
            if a is not None:
                ln_mcp_r.set_data(a[:,0], a[:,1])
                ln_mcp_s.set_data(a[:,0], a[:,2])
                ln_mcp_c.set_data(a[:,0], a[:,3])
                ax_jnt.set_xlim(max(0, a[-1,0]-WINDOW_S), a[-1,0]+0.3)
                all_z = list(a[:,1:4].ravel())
                for buf2, ln in [(pd, ln_pip_r), (dd, ln_dip_r), (td, ln_tip_r)]:
                    b = _trim(buf2)
                    if b is not None:
                        ln.set_data(b[:,0], b[:,1])
                        all_z.extend(b[:,1].tolist())
                if all_z:
                    mn, mx = min(all_z), max(all_z)
                    pad = max((mx-mn)*0.2, 0.02)
                    ax_jnt.set_ylim(mn-pad, mx+pad)

            # --- pointing U & V ---
            a = _trim(ud)
            if a is not None:
                ln_ur.set_data(a[:,0], a[:,1]); ln_us.set_data(a[:,0], a[:,2])
                ln_uc.set_data(a[:,0], a[:,3])
                ln_vr.set_data(a[:,0], a[:,4]); ln_vs.set_data(a[:,0], a[:,5])
                ln_vc.set_data(a[:,0], a[:,6])
                for ax in (ax_u, ax_v):
                    ax.set_xlim(max(0, a[-1,0]-WINDOW_S), a[-1,0]+0.3)

        ani = animation.FuncAnimation(fig, update, interval=UPDATE_MS, blit=False)
        self._ani = ani
        plt.show()


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    tuner = OEFTuner()
    spin_thread = threading.Thread(target=rospy.spin, daemon=True)
    spin_thread.start()
    tuner.build_gui()
