# Architecture

The system is organized as a ROS 1 workspace with separate packages for sensing, perception, calibration, pointing localization, visualization, and robot control.

## Pipeline

```text
D435i camera
  -> /d435i/rgb/image_raw
  -> /d435i/depth/image_raw

hand_detection
  -> /hand/detection

finger_analysis
  -> /finger/analysis

camera_calibration
  -> /camera/calibration

pointing_localization
  -> /pointing/target

sphero_detection
  -> /sphero/detection

sphero_pointing
  -> Sphero heading/speed commands

visualization
  -> live RGB and top-down workspace views
```

## Main Packages

| Package | Purpose |
| --- | --- |
| `d435i_camera` | Streams aligned RGB and depth frames from the RealSense D435i |
| `hand_detection` | Detects MediaPipe hand landmarks and publishes custom hand messages |
| `finger_analysis` | Reconstructs index finger joints in 3D and validates pointing straightness |
| `camera_calibration` | Builds the workspace plane from four clicked paper corners |
| `pointing_localization` | Intersects the 3D pointing ray with the calibrated workspace plane |
| `sphero_detection` | Detects the Sphero position in the rectified workspace view |
| `sphero_pointing` | Converts pointing targets into closed-loop Sphero navigation commands |
| `visualization` | Displays system state and workspace feedback |

## Coordinate Frames

The RealSense camera provides RGB and depth frames. Hand landmarks are detected in image coordinates, paired with depth, and back-projected into camera-space 3D points. The workspace calibration defines a planar 850 mm x 600 mm coordinate system from the paper corners. Pointing targets and Sphero positions are converted into the same workspace coordinates so navigation error can be measured directly.

## Launch Entry Point

The main launch file is:

```bash
roslaunch visualization thesis_system.launch
```

Most tunable parameters, including camera intrinsics, paper dimensions, filter values, thresholds, and Sphero settings, are configured in [`thesis_ws/src/visualization/launch/thesis_system.launch`](../thesis_ws/src/visualization/launch/thesis_system.launch).
