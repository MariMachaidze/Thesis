# Thesis System: Hand Pointing Localization with D435i Camera

A ROS 1 system for detecting hand gestures, analyzing finger straightness, and localizing pointing targets on a reference paper plane using the Intel RealSense D435i camera.

## Project Structure

```
thesis_ws/
├── src/
│   ├── d435i_camera/          # Camera driver (RGB + Depth streaming)
│   ├── hand_detection/        # Hand landmark detection (MediaPipe)
│   ├── finger_analysis/       # Finger straightness analysis (3D)
│   ├── camera_calibration/    # White paper plane calibration
│   ├── pointing_localization/ # Ray-plane intersection for pointing target
│   └── visualization/         # Real-time visualization & launch files
└── README.md
```

## System Architecture

The system is divided into **5 independent ROS packages** that communicate via topics:

```
D435i Camera
    ↓
[d435i_camera] → /d435i/rgb/image_raw, /d435i/depth/image_raw
    ↓
    ├→ [hand_detection] → /hand/detection (Hand message)
    ├→ [camera_calibration] → /camera/calibration (CalibrationData message)
    │
    ├→ [finger_analysis] → /finger/analysis (FingerAnalysis message)
    │
    └→ [pointing_localization] → /pointing/target (PointingTarget message)
    
    ↓
[visualization] ← subscribes to all topics above
```

## Data Flow

1. **Camera Driver** (`d435i_camera`):
   - Initializes RealSense D435i
   - Publishes RGB and Depth frames at 30 FPS
   - Provides camera intrinsics

2. **Hand Detection** (`hand_detection`):
   - Uses MediaPipe to detect 21 hand landmarks
   - Publishes hand keypoints in normalized image coordinates (0-1)

3. **Camera Calibration** (`camera_calibration`):
   - Interactive: user clicks 4 corners of white paper
   - Computes 3D plane equation of paper
   - Calculates camera tilt angle
   - Stores calibration data for other nodes

4. **Finger Analysis** (`finger_analysis`):
   - Receives hand landmarks and depth frame
   - Converts 2D image coordinates to 3D camera space using depth
   - Calculates finger straightness score (0-1)
   - Publishes 3D finger position and direction vector

5. **Pointing Localization** (`pointing_localization`):
   - Receives finger analysis and calibration data
   - Computes ray-plane intersection:
     - Ray: from knuckle through fingertip
     - Plane: paper plane from calibration
   - Projects intersection to paper 2D coordinates
   - Only publishes if finger is straight AND intersection is within paper bounds

6. **Visualization** (`visualization`):
   - Real-time RGB view with hand landmarks
   - Top-down view of paper with pointing target
   - Displays straightness score, camera tilt, and other metrics

## Installation & Setup

### Prerequisites

```bash
sudo apt-get install python3-pip
pip3 install opencv-python mediapipe pyrealsense2 numpy
```

### Build the ROS Workspace

```bash
cd ~/thesis_ws
catkin_make
source devel/setup.bash
```

### Launch the System

```bash
roslaunch visualization thesis_system.launch
```

## Custom Messages

### Hand.msg
```
Header header
geometry_msgs/Point[] keypoints_2d       # 21 hand landmarks (normalized)
float32[] confidences                    # Confidence for each landmark
string handedness                        # "Left" or "Right"
bool detected                            # Hand detected this frame
```

### FingerAnalysis.msg
```
Header header
geometry_msgs/Point knuckle_2d           # Index knuckle (normalized image)
geometry_msgs/Point tip_2d               # Index tip (normalized image)
geometry_msgs/Point knuckle_3d           # Index knuckle (meters, camera space)
geometry_msgs/Point tip_3d               # Index tip (meters, camera space)
geometry_msgs/Vector3 direction_3d       # Normalized direction (knuckle→tip)
float32 straightness_score               # 0-1, where 1 = perfectly straight
bool is_straight                         # True if > threshold
float32 confidence                       # Detection confidence
```

### CalibrationData.msg
```
Header header
geometry_msgs/Vector3 plane_normal       # Normal vector to paper plane
geometry_msgs/Point plane_center         # Center of paper (meters)
geometry_msgs/Vector3 paper_x_axis       # Paper X-axis (direction)
geometry_msgs/Vector3 paper_y_axis       # Paper Y-axis (direction)
float32 paper_width_mm                   # Paper width (mm)
float32 paper_height_mm                  # Paper height (mm)
float32 tilt_angle_rad                   # Camera tilt vs paper
float32 distance_to_paper_m              # Distance to paper (meters)
bool is_calibrated                       # Calibration valid
```

### PointingTarget.msg
```
Header header
float32 u_normalized                     # Paper X coordinate (0-1)
float32 v_normalized                     # Paper Y coordinate (0-1)
float32 u_mm                             # Paper X coordinate (mm)
float32 v_mm                             # Paper Y coordinate (mm)
float32 straightness                     # Finger straightness score
bool is_valid                            # Within paper bounds
float32 confidence                       # Overall confidence
```

## Usage

### Workflow

1. **Start the system:**
   ```bash
   roslaunch visualization thesis_system.launch
   ```

2. **Calibrate the camera** (camera_calibration window will appear):
   - Click 4 corners of your white reference paper
   - Corners: Top-Left → Top-Right → Bottom-Left → Bottom-Right
   - System computes paper plane and stores calibration

3. **Point with your hand:**
   - Show your hand to the camera
   - Extend index finger
   - Keep it relatively straight (> 70% straightness threshold)
   - Red dot appears on top-down view showing where you're pointing

4. **Monitor the outputs:**
   - **RGB Window**: Hand skeleton + finger direction line
   - **Top-Down Window**: Paper with red dot showing pointing target
   - Status shows: straightness score, camera tilt, hand detection

### ROS Topics

**Subscribed by visualization:**
- `/d435i/rgb/image_raw` (sensor_msgs/Image)
- `/hand/detection` (hand_detection/Hand)
- `/finger/analysis` (finger_analysis/FingerAnalysis)
- `/camera/calibration` (camera_calibration/CalibrationData)
- `/pointing/target` (pointing_localization/PointingTarget)

### Parameters

Edit `launch/thesis_system.launch` to adjust:
- `width`, `height`, `fps`: Camera resolution
- `straightness_threshold`: Minimum finger straightness to show target
- `detection_confidence`, `tracking_confidence`: MediaPipe sensitivity
- `paper_width_mm`, `paper_height_mm`: Reference paper dimensions

## Key Mathematics

### Straightness Calculation
Computes angles between consecutive finger segments (knuckle-PIP-DIP-TIP) and calculates how close they are to 180° (straight line).

**Formula:**
```
vec1 = PIP - MCP
vec2 = DIP - PIP  
angle = arccos(dot(vec1, vec2) / (|vec1| * |vec2|))
straightness = cos(|angle - 180°|)  ∈ [0, 1]
```

### 3D Projection
Uses camera intrinsics to convert 2D pixels to 3D points:
```
X = (x - cx) * depth / fx
Y = (y - cy) * depth / fy
Z = depth
```

### Ray-Plane Intersection
Finds where finger ray intersects paper plane:
```
Ray: P(t) = knuckle + t × direction
Plane: normal · (P - plane_center) = 0

t = normal · (plane_center - knuckle) / (normal · direction)
intersection = knuckle + t × direction
```

## Troubleshooting

### Camera Not Found
```bash
# Check USB connection
lsusb | grep RealSense

# Check /dev/video*
ls /dev/video*
```

### Nodes Not Starting
```bash
# Check ROS setup
source /opt/ros/noetic/setup.bash
cd ~/thesis_ws && source devel/setup.bash

# Check if messages compiled
rosmsg show hand_detection/Hand
```

### Poor Finger Detection
- Ensure good lighting
- Increase `detection_confidence` (more strict) or decrease (more lenient)
- Keep hand within frame

### Pointing Inaccurate
- Recalibrate by pressing 'r' in camera_calibration window
- Ensure paper is flat and well-lit
- Check camera is perpendicular to paper (optimal ~45° tilt)

## Future Improvements

- [ ] Automatic paper detection (instead of manual clicks)
- [ ] Support for both hands simultaneously
- [ ] Multi-finger analysis (thumb, middle, ring)
- [ ] RViz visualization plugin
- [ ] Dynamic finger straightness tuning via GUI
- [ ] Depth-based distance estimation
- [ ] Kalman filtering for smoother output

## Author Notes

This system was developed for thesis work on hand gesture recognition and 3D pointing localization. The modular ROS package structure allows easy integration into larger systems.

For more details, see the individual package READMEs and the main thesis document.
