#!/usr/bin/env python3
"""
test_svd_direction.py
---------------------
Standalone (no ROS) unit tests for fit_finger_direction().

Each test prints PASS or FAIL with details.
Run directly:
    python3 thesis_ws/src/finger_analysis/scripts/test_svd_direction.py
"""

import numpy as np

# ── Copy of the function under test (no ROS imports needed) ──────────────────

def fit_finger_direction(points_3d):
    valid = [p for p in points_3d if p is not None]
    if len(valid) < 2:
        return None

    pts      = np.array(valid)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    _, _, Vt  = np.linalg.svd(centered)
    direction = Vt[0]

    if points_3d[0] is not None and points_3d[-1] is not None:
        rough_dir = points_3d[-1] - points_3d[0]
    else:
        rough_dir = valid[-1] - valid[0]
    if np.dot(direction, rough_dir) < 0:
        direction = -direction

    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None
    return direction / norm


# ── Helpers ───────────────────────────────────────────────────────────────────

def check(name, result, expected, tol=0.01):
    if result is None:
        print(f"FAIL  {name}: returned None")
        return
    cos = abs(np.dot(result, expected))          # 1.0 = parallel
    angle_deg = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ok = angle_deg < np.degrees(np.arccos(1 - tol))  # ~8 deg at tol=0.01
    status = "PASS" if ok else "FAIL"
    # Also check it points the right way (not flipped)
    sign_ok = np.dot(result, expected) > 0
    if not sign_ok:
        status = "FAIL (wrong sign)"
    print(f"{status}  {name}")
    print(f"       result  = [{result[0]:+.4f}, {result[1]:+.4f}, {result[2]:+.4f}]")
    print(f"       expected= [{expected[0]:+.4f}, {expected[1]:+.4f}, {expected[2]:+.4f}]")
    print(f"       angle   = {angle_deg:.2f}°")

def check_none(name, result):
    ok = result is None
    print(f"{'PASS' if ok else 'FAIL'}  {name}: expected None, got {result}")


# ── Test cases ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("fit_finger_direction — SVD unit tests")
print("=" * 60)

# 1. Perfect straight finger pointing along +Z (depth axis)
#    MCP at 1.0m, joints evenly spaced 3 cm apart
pts_z = [
    np.array([0.0, 0.0, 1.00]),   # MCP (knuckle)
    np.array([0.0, 0.0, 1.03]),   # PIP
    np.array([0.0, 0.0, 1.06]),   # DIP
    np.array([0.0, 0.0, 1.09]),   # TIP
]
check("straight finger → +Z", fit_finger_direction(pts_z), np.array([0, 0, 1]))

# 2. Finger pointing diagonally (equal X and Z components)
diag = np.array([1, 0, 1]) / np.sqrt(2)
pts_diag = [np.array([0.0, 0.0, 0.0]) + i * 0.03 * diag for i in range(4)]
check("diagonal +X+Z finger", fit_finger_direction(pts_diag), diag)

# 3. Realistic D435i finger at ~1.5 m pointing slightly toward camera+right
#    (hand extended, finger pointing at a paper 40 cm in front)
base = np.array([0.1, -0.05, 1.50])
direction_true = np.array([-0.10, 0.05, -0.40])
direction_true /= np.linalg.norm(direction_true)
pts_real = [base + i * 0.025 * direction_true for i in range(4)]
check("realistic D435i geometry", fit_finger_direction(pts_real), direction_true)

# 4. Noisy joints — small random perturbation on each point
rng = np.random.default_rng(42)
noise_mm = rng.normal(0, 0.003, (4, 3))   # 3 mm std dev per axis
pts_noisy = [p + n for p, n in zip(pts_z, noise_mm)]
check("noisy joints (3 mm std)", fit_finger_direction(pts_noisy), np.array([0, 0, 1]), tol=0.02)

# 5. Missing TIP depth (None) — should still return direction from 3 points
pts_no_tip = [
    np.array([0.0, 0.0, 1.00]),
    np.array([0.0, 0.0, 1.03]),
    np.array([0.0, 0.0, 1.06]),
    None,                              # TIP has no depth
]
check("tip=None, 3 valid joints", fit_finger_direction(pts_no_tip), np.array([0, 0, 1]))

# 6. Missing MCP and TIP — 2 valid middle joints (minimum case)
pts_middle_only = [
    None,
    np.array([0.0, 0.0, 1.03]),
    np.array([0.0, 0.0, 1.06]),
    None,
]
check("only PIP+DIP valid (2 pts)", fit_finger_direction(pts_middle_only), np.array([0, 0, 1]))

# 7. Only 1 valid point — must return None
pts_one = [None, None, np.array([0.0, 0.0, 1.03]), None]
check_none("only 1 valid point → None", fit_finger_direction(pts_one))

# 8. All None — must return None
check_none("all None → None", fit_finger_direction([None, None, None, None]))

# 9. Sign check: finger pointing in -Z (away from wall, toward camera)
#    This is unnatural but SVD might give +Z; sign flip should catch it.
pts_neg_z = [
    np.array([0.0, 0.0, 1.09]),   # MCP far
    np.array([0.0, 0.0, 1.06]),
    np.array([0.0, 0.0, 1.03]),
    np.array([0.0, 0.0, 1.00]),   # TIP close
]
check("sign flip: MCP far, TIP close → -Z", fit_finger_direction(pts_neg_z), np.array([0, 0, -1]))

print("=" * 60)
