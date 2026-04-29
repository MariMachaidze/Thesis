#!/usr/bin/env python3
"""
ball_tracker.py — Track a single Sphero Bolt on blue carpet using Hough circles.
Uses Intel RealSense D435i RGB stream.

Detection strategy:
  - Grayscale → CLAHE (local contrast boost) → Gaussian blur → HoughCircles
  - When multiple circles found, pick the one closest to the last known position
  - Lost-frame buffer: trail stays visible for a few missed frames before fading

Keys:
  Q — quit
  R — redraw ROI

Usage:
  python3 ball_tracker.py
"""

import cv2
import numpy as np
import pyrealsense2 as rs

# ── Camera ────────────────────────────────────────────────────────────────────
WIDTH  = 1280
HEIGHT = 720
FPS    = 30

# ── Hough defaults (tunable via trackbars) ────────────────────────────────────
# At ~3 m, Sphero (7.5 cm diam) ≈ 10-15 px radius in frame.
DEF_PARAM2 = 20    # accumulator threshold — lower = more sensitive
DEF_MIN_R  = 6    # px
DEF_MAX_R  = 40   # px
DEF_BLUR   = 3    # Gaussian blur kernel (odd number)

HOUGH_DP      = 1.2
HOUGH_MINDIST = 40     # px — min distance between two circle centers
HOUGH_PARAM1  = 80     # Canny upper threshold inside Hough

# ── Tracking ──────────────────────────────────────────────────────────────────
TRAIL_LEN  = 50    # max positions to draw
MAX_LOST   = 8     # frames to keep last position before declaring lost
MAX_JUMP   = 80    # px — max displacement between frames (ignores teleporting noise)

# ── CLAHE ─────────────────────────────────────────────────────────────────────
CLAHE_CLIP  = 2.0
CLAHE_GRID  = (8, 8)
# ─────────────────────────────────────────────────────────────────────────────

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)


def nothing(_):
    pass


# ── RealSense ─────────────────────────────────────────────────────────────────

def start_pipeline():
    p   = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    p.start(cfg)
    return p


def get_frame(pipeline):
    frames = pipeline.wait_for_frames(timeout_ms=5000)
    cf = frames.get_color_frame()
    return np.asanyarray(cf.get_data()) if cf else None


# ── ROI selection ─────────────────────────────────────────────────────────────

def select_roi(pipeline):
    print("\nDrag ROI over the ground, then SPACE/ENTER. Press C for full frame.\n")
    frame = None
    while frame is None:
        frame = get_frame(pipeline)

    snap = frame.copy()
    cv2.putText(snap, "Drag ROI  |  SPACE/ENTER = confirm  |  C = full frame",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI", 1280, 720)
    rect = cv2.selectROI("Select ROI", snap, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    rx, ry, rw, rh = rect
    if rw < 20 or rh < 20:
        x1, y1, x2, y2 = 0, 0, WIDTH, HEIGHT
        print("Using full frame.")
    else:
        x1, y1, x2, y2 = rx, ry, rx + rw, ry + rh
        print(f"ROI: ({x1},{y1}) → ({x2},{y2})")
    return x1, y1, x2, y2


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_hough(frame_bgr, x1, y1, x2, y2, p2, min_r, max_r, blur_k):
    """
    Run Hough circle detection on the ROI crop.
    Returns list of (cx, cy, r) in full-frame coords (may be empty).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    roi  = gray[y1:y2, x1:x2]

    # CLAHE: boost local contrast so the ball's highlight/edge stands out
    enhanced = _clahe.apply(roi)

    k    = max(3, blur_k | 1)   # ensure odd
    blur = cv2.GaussianBlur(enhanced, (k, k), 0)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp        = HOUGH_DP,
        minDist   = HOUGH_MINDIST,
        param1    = HOUGH_PARAM1,
        param2    = max(1, p2),
        minRadius = max(1, min_r),
        maxRadius = max(2, max_r),
    )
    if circles is None:
        return []

    results = []
    for cx_r, cy_r, r in np.round(circles[0]).astype(int):
        results.append((x1 + int(cx_r), y1 + int(cy_r), int(r)))
    return results


def pick_best(candidates, last_pos):
    """
    From a list of (cx, cy, r) candidates:
      - If we have a last known position, prefer the closest within MAX_JUMP.
      - Otherwise return the first (strongest from Hough).
    """
    if not candidates:
        return None
    if last_pos is None:
        return candidates[0]

    lx, ly = last_pos
    best, best_dist = None, float('inf')
    for cx, cy, r in candidates:
        d = np.hypot(cx - lx, cy - ly)
        if d < best_dist and d < MAX_JUMP:
            best_dist = d
            best      = (cx, cy, r)
    # If nothing within MAX_JUMP, fall back to strongest
    return best if best is not None else candidates[0]


# ── Trail ─────────────────────────────────────────────────────────────────────

def draw_trail(img, trail):
    n = len(trail)
    for i in range(1, n):
        a   = i / n
        col = (int(50 * a), int(200 * a), int(255 * (1 - a) + 80 * a))
        cv2.line(img, trail[i - 1], trail[i], col, max(1, int(3 * a)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Starting RealSense D435i...")
    pipeline = start_pipeline()

    x1, y1, x2, y2 = select_roi(pipeline)

    # ── Windows & trackbars ───────────────────────────────────────────────────
    WIN    = "Ball Tracker  [Q=quit  R=redraw ROI]"
    WIN_DBG= "Hough input (ROI, CLAHE)"
    WIN_TB = "Hough Controls"

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.namedWindow(WIN_DBG, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_DBG, max(1, x2 - x1), max(1, y2 - y1))
    cv2.namedWindow(WIN_TB, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("Sensitivity (p2)\nlower=more",  WIN_TB, DEF_PARAM2, 80,  nothing)
    cv2.createTrackbar("Min radius (px)",               WIN_TB, DEF_MIN_R,  100, nothing)
    cv2.createTrackbar("Max radius (px)",               WIN_TB, DEF_MAX_R,  200, nothing)
    cv2.createTrackbar("Blur kernel (px)",              WIN_TB, DEF_BLUR,   15,  nothing)

    # ── State ─────────────────────────────────────────────────────────────────
    trail     = []
    last_pos  = None   # (cx, cy) of last confirmed detection
    lost_count= 0      # consecutive frames without detection

    print("\nTracking live.")
    print("  Adjust 'Hough Controls' window to tune detection.")
    print("  Q = quit   R = redraw ROI\n")

    while True:
        frame = get_frame(pipeline)
        if frame is None:
            continue

        p2     = cv2.getTrackbarPos("Sensitivity (p2)\nlower=more",  WIN_TB)
        min_r  = cv2.getTrackbarPos("Min radius (px)",               WIN_TB)
        max_r  = cv2.getTrackbarPos("Max radius (px)",               WIN_TB)
        blur_k = cv2.getTrackbarPos("Blur kernel (px)",              WIN_TB)

        # ── Detect ────────────────────────────────────────────────────────────
        candidates = detect_hough(frame, x1, y1, x2, y2, p2, min_r, max_r, blur_k)
        detected   = pick_best(candidates, last_pos)

        if detected is not None:
            cx, cy, r  = detected
            last_pos   = (cx, cy)
            lost_count = 0
            trail.append((cx, cy))
            if len(trail) > TRAIL_LEN:
                trail.pop(0)
        else:
            lost_count += 1
            if lost_count > MAX_LOST:
                last_pos = None
                if trail:
                    trail.pop(0)   # fade trail slowly

        # ── Draw main window ──────────────────────────────────────────────────
        disp = frame.copy()
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_trail(disp, trail)

        # Draw all candidates faintly
        for cx_c, cy_c, r_c in candidates:
            cv2.circle(disp, (cx_c, cy_c), r_c, (80, 80, 80), 1)

        if detected is not None:
            cx, cy, r = detected
            cv2.circle(disp, (cx, cy), max(r, 4), (0, 255, 255), 2)   # ball ring
            cv2.circle(disp, (cx, cy), 4, (0, 0, 255), -1)             # center dot
            cv2.putText(disp, f"({cx}, {cy})  r={r}px",
                        (cx + r + 6, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            status = f"searching...  (lost {lost_count}f)" if lost_count > 0 else "searching..."
            cv2.putText(disp, status, (x1 + 6, y1 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 1)

        cv2.putText(disp,
                    f"p2={p2}  r={min_r}-{max_r}px  blur={max(3,blur_k|1)}  |  Q=quit  R=ROI",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1)

        cv2.imshow(WIN, disp)

        # ── Debug window: what Hough sees (CLAHE ROI) ─────────────────────────
        gray_roi  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2]
        clahe_roi = _clahe.apply(gray_roi)
        dbg       = cv2.cvtColor(clahe_roi, cv2.COLOR_GRAY2BGR)
        # Mark detected circle on debug view too
        if detected is not None:
            cx, cy, r = detected
            cv2.circle(dbg, (cx - x1, cy - y1), max(r, 4), (0, 255, 255), 2)
            cv2.circle(dbg, (cx - x1, cy - y1), 3, (0, 0, 255), -1)
        cv2.imshow(WIN_DBG, dbg)

        # ── Keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            cv2.destroyWindow(WIN_DBG)
            x1, y1, x2, y2 = select_roi(pipeline)
            cv2.namedWindow(WIN_DBG, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_DBG, max(1, x2 - x1), max(1, y2 - y1))
            trail.clear()
            last_pos   = None
            lost_count = 0

    cv2.destroyAllWindows()
    pipeline.stop()
    print("Done.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nInterrupted.")
