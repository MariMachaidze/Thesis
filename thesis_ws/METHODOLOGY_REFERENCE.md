# Thesis System — Methodology Reference

Generated from source code analysis of `thesis_ws/src/`. Covers all 8 ROS packages, 17 nodes.

---

## 1. System Overview

The system enables a user to point a finger at a paper target, detects the pointing direction in 3D, and commands a Sphero BOLT robot to navigate to the pointed location. The pipeline is fully vision-based using an Intel RealSense D435i depth camera.

**Pipeline summary:**

```
RGB + Depth frames (D435i)
    ↓
Hand Landmark Detection (MediaPipe)
    ↓
Finger 3D Position + Straightness Analysis
    ↓
Paper Plane Calibration + Ray-Plane Intersection
    ↓
Pointing Target (u, v on paper, in mm)
    ↓
Sphero Navigation (closed-loop)
```

---

## 2. Hardware

| Component | Specification |
|-----------|--------------|
| Camera | Intel RealSense D435i |
| Resolution | 1280 × 720 pixels |
| Frame rate | 30 Hz |
| Focal length | fx = 901.473 px, fy = 899.637 px |
| Principal point | cx = 642.351 px, cy = 349.990 px |
| Depth scale | 0.001 m/unit (raw → metres) |
| Robot | Sphero BOLT |
| Target surface | A0 paper sheet, 850 mm × 600 mm |

---

## 3. ROS Node Graph

```
/d435i_driver_node
    → /d435i/rgb/image_raw
    → /d435i/depth/image_raw

/hand_detection_node  ← /d435i/rgb/image_raw
    → /hand/detection

/camera_calibration_node  ← /d435i/rgb/image_raw, /d435i/depth/image_raw
    → /camera/calibration  (latched)

/finger_analysis_node  ← /hand/detection, /d435i/depth/image_raw
    → /finger/analysis
    → /diag/straightness_raw, /diag/knuckle_raw, /diag/pip_raw, /diag/dip_raw, /diag/tip_raw

/pointing_localization_node  ← /finger/analysis, /camera/calibration
    → /pointing/target

/sphero_detection_node  ← /d435i/rgb/image_raw, /camera/calibration
    → /sphero/detection

/sphero_navigate_node  ← /pointing/target, /sphero/detection, /d435i/rgb/image_raw, /camera/calibration
    → /sphero/travel_time_s

/visualization_node  ← all above topics
```

---

## 4. Package-by-Package Description

---

### 4.1 `d435i_camera` — Camera Driver

**Node:** `d435i_driver_node.py`

Interfaces with the Intel RealSense D435i via the `pyrealsense2` SDK. Enables depth-RGB alignment so each depth pixel corresponds to the same physical point as the RGB pixel at the same index.

**Published topics:**
- `/d435i/rgb/image_raw` — BGR8 colour frame
- `/d435i/depth/image_raw` — mono16 depth frame (units: mm as raw integers)

**Optional post-processing:** spatial hole-filling filter (3 modes: `fill_from_left`, `farthest_from_around`, `nearest_from_around`).

---

### 4.2 `hand_detection` — Hand Landmark Detection

**Node:** `hand_detection_node.py`

**Model:** Google MediaPipe Hands (pre-trained CNN), single-hand mode.

- Outputs 21 keypoints per hand as normalized (x, y) ∈ [0, 1] coordinates.
- Relevant keypoint indices:
  - 0: Wrist
  - 5 (MCP), 6 (PIP), 7 (DIP), 8 (TIP): Index finger joints
  - 17–20: Pinky finger joints

**Architecture:** Worker thread processes frames asynchronously; ROS callback places newest frame in a shared slot (prevents double-processing under slow inference).

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `detection_confidence` | 0.65 |
| `tracking_confidence` | 0.45 |
| `max_num_hands` | 1 |
| `model_complexity` | 0 (lightweight) |

---

### 4.3 `camera_calibration` — Paper Plane Calibration

**Node:** `camera_calibration_node.py`

Determines the 3D position and orientation of the paper surface relative to the camera. Operator clicks four corners of the paper (TL, TR, BL, BR) in the RGB image.

#### 4.3.1 Back-projection (Pixel → 3D)

For each clicked pixel (u, v) with depth d (in metres):

$$X = \frac{(u - c_x) \cdot d}{f_x}, \quad Y = \frac{(v - c_y) \cdot d}{f_y}, \quad Z = d$$

Depth at each corner is extracted as the **median of a 5 × 5 pixel patch** to reduce sensor noise.

#### 4.3.2 Plane Normal via SVD

Let $\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3, \mathbf{p}_4 \in \mathbb{R}^3$ be the back-projected corner points. Centre the points:

$$\bar{\mathbf{p}} = \frac{1}{4} \sum_{i=1}^{4} \mathbf{p}_i, \quad \mathbf{q}_i = \mathbf{p}_i - \bar{\mathbf{p}}$$

Perform SVD on the matrix $Q = [\mathbf{q}_1 \; \mathbf{q}_2 \; \mathbf{q}_3 \; \mathbf{q}_4]^T$:

$$Q = U \Sigma V^T$$

The plane normal $\hat{\mathbf{n}}$ is the **last row of** $V^T$ (corresponding to the smallest singular value). The sign is flipped if $\hat{\mathbf{n}} \cdot [0,0,1] < 0$ to ensure the normal points toward the camera.

#### 4.3.3 Paper Axes

$$\hat{\mathbf{x}} = \frac{\mathbf{p}_2 - \mathbf{p}_1}{\|\mathbf{p}_2 - \mathbf{p}_1\|} \quad \text{(TL → TR)}$$

$$\hat{\mathbf{y}} = \frac{\mathbf{p}_3 - \mathbf{p}_1}{\|\mathbf{p}_3 - \mathbf{p}_1\|} \quad \text{(TL → BL)}$$

#### 4.3.4 Tilt Angle

$$\theta = \arccos\!\left(\hat{\mathbf{n}} \cdot [0,0,1]^T\right)$$

**Output** (published as `/camera/calibration`, latched):

| Field | Description |
|-------|-------------|
| `plane_normal` | Unit normal $\hat{\mathbf{n}}$ |
| `plane_center` | $\mathbf{p}_1$ (TL corner) |
| `paper_x_axis` | $\hat{\mathbf{x}}$ |
| `paper_y_axis` | $\hat{\mathbf{y}}$ |
| `paper_x_m` | 0.850 m |
| `paper_y_m` | 0.600 m |
| `tilt_angle_rad` | $\theta$ |
| `distance_to_paper_m` | Mean depth to corners |

Calibration is persisted to `~/.ros/paper_calibration.yaml`.

---

### 4.4 `finger_analysis` — Finger 3D Analysis

**Node:** `finger_analysis_node.py`

Reconstructs 3D positions of the four index finger joints and computes a **straightness score** used as a gating signal for pointing.

#### 4.4.1 Depth Extraction and Back-projection

For each joint keypoint $(u, v)$ (in pixels):

1. Extract depth as **median of a 3 × 3 pixel patch** centred on $(u, v)$ from the depth image.
2. Back-project to 3D using camera intrinsics (same equation as §4.3.1).

A **rolling median filter** (window = 5 frames) is applied independently to each of the three spatial axes (X, Y, Z) for each of the four joints. This rejects transient depth outliers; a new value must appear in the majority of the window to affect the output.

#### 4.4.2 Straightness Score

From the four 3D joint positions $\mathbf{j}_0$ (MCP), $\mathbf{j}_1$ (PIP), $\mathbf{j}_2$ (DIP), $\mathbf{j}_3$ (TIP), form three segment vectors:

$$\mathbf{s}_k = \mathbf{j}_{k+1} - \mathbf{j}_k, \quad k = 0, 1, 2$$

The alignment of consecutive segments is measured by their cosine similarity:

$$c_k = \frac{\mathbf{s}_k \cdot \mathbf{s}_{k+1}}{\|\mathbf{s}_k\| \cdot \|\mathbf{s}_{k+1}\|}, \quad k = 0, 1$$

Straightness score:

$$S = \text{clip}\!\left(\frac{c_0 + c_1}{2},\; 0,\; 1\right)$$

$S = 1$ means the finger is perfectly straight; $S = 0$ means segments are perpendicular.

#### 4.4.3 Straightness Filtering (One Euro Filter)

The raw straightness score $S$ is smoothed with a **One Euro Filter** (Casiez et al., 2012). The filter adapts its cutoff frequency based on the speed of change, reducing lag during fast motion and noise during slow motion.

**Derivative estimate** (first-order finite difference):

$$\dot{x}_n = \frac{x_n - x_{n-1}}{\Delta t}$$

**Adaptive cutoff frequency:**

$$f_c = f_{c,\min} + \beta \cdot |\dot{x}_n|$$

where $f_{c,\min} = 1.0$ Hz and $\beta = 0.007$.

**Time constant:**

$$\tau = \frac{1}{2\pi f_c}$$

**Smoothing factor:**

$$\alpha = \frac{1}{1 + \tau / \Delta t}$$

**Filtered output:**

$$\hat{x}_n = \alpha \cdot x_n + (1 - \alpha) \cdot \hat{x}_{n-1}$$

The same filter structure is applied to the derivative estimate with fixed cutoff $f_{c,d} = 1.0$ Hz.

#### 4.4.4 Hysteresis Gating

To avoid rapid toggling at the threshold:

| Condition | Transition |
|-----------|-----------|
| $\hat{S} \geq 0.65$ | → `IS_STRAIGHT = True` |
| $\hat{S} \leq 0.45$ | → `IS_STRAIGHT = False` |

#### 4.4.5 Finger Direction via SVD

When the finger is straight, a best-fit 3D line is computed through the four joint positions using SVD:

1. Compute centroid $\bar{\mathbf{j}}$.
2. Form matrix of centred joints.
3. SVD: first right singular vector = direction of maximum variance = finger axis $\hat{\mathbf{d}}$.
4. Flip sign if $\hat{\mathbf{d}}$ points toward the knuckle rather than the tip.

The pointing ray is then $\mathbf{r}(t) = \mathbf{j}_{\text{MCP}} + t\,\hat{\mathbf{d}}$.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `straight_on_threshold` | 0.65 |
| `straight_off_threshold` | 0.45 |
| OEF `min_cutoff` | 1.0 Hz |
| OEF `beta` | 0.007 |
| `joint_median_window` | 5 frames |
| `depth_patch_size` | 3 px |
| `hand_loss_reset_frames` | 5 |

---

### 4.5 `pointing_localization` — Ray-Plane Intersection

**Node:** `pointing_localization_node.py`

Determines where the pointing ray intersects the paper plane, producing a 2D coordinate on the paper surface.

#### 4.5.1 Ray-Plane Intersection

The pointing ray (from finger analysis):

$$\mathbf{r}(t) = \mathbf{o} + t\,\hat{\mathbf{d}}$$

where $\mathbf{o}$ is the ray origin (MCP joint position) and $\hat{\mathbf{d}}$ is the unit direction.

The paper plane satisfies:

$$\hat{\mathbf{n}} \cdot (\mathbf{x} - \mathbf{p}_0) = 0$$

Substituting the ray equation and solving for $t$:

$$t = \frac{\hat{\mathbf{n}} \cdot (\mathbf{p}_0 - \mathbf{o})}{\hat{\mathbf{n}} \cdot \hat{\mathbf{d}}}$$

The 3D intersection point:

$$\mathbf{x}^* = \mathbf{o} + t\,\hat{\mathbf{d}}$$

Only valid when $|\hat{\mathbf{n}} \cdot \hat{\mathbf{d}}| > \epsilon$ (ray not parallel to plane) and $t > 0$ (intersection is in front of camera).

#### 4.5.2 Projection to Paper Coordinates

Vector from paper origin $\mathbf{p}_0$ (TL corner) to intersection:

$$\mathbf{v} = \mathbf{x}^* - \mathbf{p}_0$$

Normalized paper coordinates:

$$u = \frac{\mathbf{v} \cdot \hat{\mathbf{x}}}{L_x}, \quad v = \frac{\mathbf{v} \cdot \hat{\mathbf{y}}}{L_y}$$

where $L_x = 0.850$ m and $L_y = 0.600$ m. In millimetres: $x_{\text{mm}} = u \cdot L_x \cdot 1000$, $y_{\text{mm}} = v \cdot L_y \cdot 1000$.

Valid only when $u \in [0, 1]$ and $v \in [0, 1]$ (intersection is on the paper).

#### 4.5.3 Output Filtering (One Euro Filter)

Separate One Euro Filters are applied to $u$ and $v$ with the same parameters as §4.4.3. The filter state resets if no valid pointing data arrives for more than 0.5 s (prevents large transient lag on resumption).

**Parameters:**

| Parameter | Value |
|-----------|-------|
| OEF `min_cutoff` | 1.0 Hz |
| OEF `beta` | 0.007 |
| `uv_gap_reset_s` | 0.5 s |
| NOT_STRAIGHT reset | 5 frames |

---

### 4.6 `sphero_detection` — Robot Detection

**Node:** `sphero_detection_node.py`

Detects the Sphero BOLT in the camera image using its circular silhouette and RGB LED.

#### 4.6.1 Pre-processing

1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization): applied to the luminance channel to enhance local contrast. `clipLimit = 0.6`.
2. **Gaussian blur**: kernel size 6 (rounded to odd) to suppress high-frequency noise before circle detection.

#### 4.6.2 Hough Circle Detection

Applies OpenCV's `HoughCircles` with the `HOUGH_GRADIENT` method:

$$\text{HoughCircles}(I, \text{dp}=1.2,\; d_{\min}=40,\; p_1=80,\; p_2=21,\; r_{\min}=25,\; r_{\max}=40)$$

- `dp`: inverse ratio of accumulator resolution to image resolution.
- $d_{\min}$: minimum distance between circle centres (px).
- $p_1$: upper threshold for Canny edge detector.
- $p_2$: accumulator threshold for circle centres (lower → more detections).
- $r_{\min}, r_{\max}$: search radius range (px).

#### 4.6.3 Perspective Rectification

The detected centre in camera pixels is transformed to the top-down paper coordinate frame via a **homography** $H$ (computed from calibration corner points):

$$\begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} \sim H \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$

Output space: 850 × 600 px (1 px = 1 mm).

#### 4.6.4 Candidate Filtering and Tracking

- **Spatial gating:** rejects detections with Euclidean distance > 120 px from last confirmed position.
- **Lost frame tolerance:** up to 10 consecutive missing frames before track is reset.
- **Trail buffer:** deque of last 80 positions for visualisation.

#### 4.6.5 One Euro Filter (Position)

Separate One Euro Filters on $u$ and $v$ with update rate 30 Hz, `min_cutoff = 0.6`, `beta = 0.03`.

**Parameters:**

| Parameter | Default |
|-----------|---------|
| Hough `p2` | 21 |
| `min_r` | 25 px |
| `max_r` | 40 px |
| Gaussian kernel | 6 |
| CLAHE `clipLimit` | 0.6 |
| OEF `min_cutoff` | 0.6 |
| OEF `beta` | 0.03 |
| Max jump | 120 px |
| Max lost frames | 10 |

---

### 4.7 `sphero_pointing` / `sphero_navigate` — Robot Navigation

**Node:** `sphero_navigate_node.py`

Closed-loop controller that steers the Sphero BOLT to the pointing target.

#### 4.7.1 Perspective Warp

Camera RGB frame is rectified to the 850 × 600 mm top-down view using a **4-point homography** from calibration data (same $H$ as §4.6.3):

$$\text{dst} = \{[0,0],\; [850,0],\; [0,600],\; [850,600]\}$$

#### 4.7.2 Heading Calculation

Given robot position $(u_r, v_r)$ and target $(u_t, v_t)$ in the rectified frame (mm):

$$\Delta u = u_t - u_r, \quad \Delta v = v_t - v_r$$

$$\theta = \arctan2(\Delta u,\; \Delta v) \cdot \frac{180}{\pi}$$

Note: $\arctan2(\Delta u, \Delta v)$ rather than the standard $\arctan2(\Delta y, \Delta x)$ because the Sphero heading convention aligns 0° with the positive $v$ (down) axis.

#### 4.7.3 Speed Scaling

$$d = \sqrt{(\Delta u)^2 + (\Delta v)^2}$$

$$\text{speed} = \max\!\left(\text{SPEED}_{\min},\; \text{SPEED} \cdot \min\!\left(\frac{d}{d_{\text{slow}}}, 1.0\right)\right)$$

This linearly ramps down speed for $d < d_{\text{slow}}$ to prevent overshoot.

#### 4.7.4 Stuck Detection

If displacement over the last `STUCK_TIMEOUT_S` = 2.5 s is < `STUCK_MOVE_CM` = 1.5 cm, the robot is declared stuck and enters a hover state. Navigation resumes when distance to target > `HOVER_RESUME_CM` = 9 cm.

#### 4.7.5 Auto-heading Calibration (sphero_pointing_node)

Used in single-shot mode to align the Sphero's internal heading reference with the camera frame:

1. Roll at heading = 0° for a fixed duration.
2. Measure actual displacement vector $(\Delta u, \Delta v)$ in the camera frame.
3. Compute calibration offset:

$$\phi_{\text{offset}} = \arctan2(\Delta u,\; -\Delta v) \cdot \frac{180}{\pi}$$

4. All subsequent heading commands are offset by $\phi_{\text{offset}}$.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `SPEED` | 50 |
| `MIN_SPEED` | 25 |
| `UPDATE_INTERVAL` | 0.3 s |
| `NEAR_DIST_CM` | 20 cm |
| `SLOW_DIST_CM` | 20 cm |
| `HOVER_STOP_CM` | 3 cm |
| `HOVER_RESUME_CM` | 9 cm |
| `COAST_FACTOR` | 0.28 |
| `STUCK_TIMEOUT_S` | 2.5 s |
| `BORDER_CM` | 5 cm |

---

### 4.8 `visualization` — Display Node

**Node:** `visualization_node.py`

Renders two live views:
1. **Hand view** (1280 × 720): camera feed with hand skeleton, pointing ray projected from finger joints.
2. **Top-down view** (850 × 600): perspective-corrected overhead view of the paper with Sphero trail and pointing target.

The pointing ray is projected back to camera pixels by applying the inverse homography $H^{-1}$ to the paper (u, v) coordinate.

---

## 5. Coordinate Systems

| Frame | Origin | X-axis | Y-axis | Z-axis | Units |
|-------|--------|--------|--------|--------|-------|
| Camera (D435i) | Camera optical centre | Right | Down | Forward (depth) | metres |
| Pixel | Top-left of image | Right | Down | — | pixels |
| Paper (3D) | Top-left corner (TL) | TL → TR | TL → BL | Plane normal | metres |
| Paper (normalised) | TL | Along $\hat{\mathbf{x}}$ | Along $\hat{\mathbf{y}}$ | — | [0, 1] |
| Rectified (warp) | Top-left | Right | Down | — | px (= mm) |

### Full transformation chain

```
MediaPipe (norm. pixel)
    → pixel (u·W, v·H)
    → 3D camera frame  [back-projection, §4.3.1]
    → paper 3D          [ray-plane intersection, §4.5.1]
    → paper (u, v)      [axis projection, §4.5.2]
    → One Euro filtered (u, v)
    → rectified px      [homography H, §4.6.3]
    → Sphero heading    [atan2, §4.7.2]
```

---

## 6. Algorithms Summary

| Algorithm | Package | Purpose |
|-----------|---------|---------|
| MediaPipe Hands CNN | `hand_detection` | 21-point hand skeleton |
| Back-projection (pinhole model) | `camera_calibration`, `finger_analysis` | Pixel + depth → 3D |
| SVD plane fitting | `camera_calibration` | Robust plane normal from 4 noisy points |
| SVD line fitting | `finger_analysis` | Best-fit 3D finger axis |
| Cosine-similarity straightness | `finger_analysis` | Scalar pointing quality gate |
| Hysteresis thresholding | `finger_analysis` | Noise-robust state transitions |
| Rolling median filter | `finger_analysis` | Per-joint depth spike rejection |
| **One Euro Filter** | `finger_analysis`, `pointing_localization`, `sphero_detection` | Adaptive low-pass smoothing |
| Ray-plane intersection | `pointing_localization` | 3D → 2D paper coordinate |
| CLAHE | `sphero_detection` | Local contrast enhancement |
| Hough Circle Transform | `sphero_detection` | Circle detection (Sphero) |
| 4-point homography (perspective warp) | `sphero_detection`, `sphero_navigate`, `visualization` | Camera → top-down view |
| P-control with speed ramp | `sphero_navigate` | Smooth target approach |
| Stuck detection | `sphero_navigate` | Dead-reckoning failure recovery |
| Heading auto-calibration | `sphero_pointing` | Camera–robot frame alignment |

---

## 7. One Euro Filter — Full Equations

Referenced in three packages (same implementation). Source: Casiez, G., Roussel, N., Vogel, D. (2012). *1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems.* CHI 2012.

Given sample $x_n$ at time $t_n$ with $\Delta t = t_n - t_{n-1}$:

**Step 1 — Derivative filter** (fixed cutoff $f_{c,d}$):

$$\tau_d = \frac{1}{2\pi f_{c,d}}, \quad \alpha_d = \frac{1}{1 + \tau_d / \Delta t}$$

$$\hat{\dot{x}}_n = \alpha_d \cdot \frac{x_n - \hat{x}_{n-1}}{\Delta t} + (1 - \alpha_d) \cdot \hat{\dot{x}}_{n-1}$$

**Step 2 — Adaptive cutoff:**

$$f_c = f_{c,\min} + \beta \cdot |\hat{\dot{x}}_n|$$

**Step 3 — Signal filter:**

$$\tau = \frac{1}{2\pi f_c}, \quad \alpha = \frac{1}{1 + \tau / \Delta t}$$

$$\hat{x}_n = \alpha \cdot x_n + (1 - \alpha) \cdot \hat{x}_{n-1}$$

**Parameters used in this system:**

| Location | $f_{c,\min}$ | $\beta$ | $f_{c,d}$ |
|----------|-------------|---------|-----------|
| Straightness score | 1.0 Hz | 0.007 | 1.0 Hz |
| Pointing (u, v) | 1.0 Hz | 0.007 | 1.0 Hz |
| Sphero position | 0.6 Hz | 0.03 | 1.0 Hz |

---

## 8. Evaluation Protocol (`pointing_accuracy_eval.py`)

- **Gesture sequence:** extend → point → open hand (triggers record start/stop).
- **Test grid:** 5 pre-defined targets × 3 heights (0, 25, 50 cm above paper) × 5 trials = 75 samples.
- **Trimming:** first 8 and last 5 samples per trial discarded (hand settling artefacts).
- **Metrics computed:** mean error (mm), standard deviation (mm), response latency (ms) per trial.
- Error = Euclidean distance between pointed coordinate and ground-truth target in mm.
