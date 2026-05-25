# Setup

This project is a ROS 1 Noetic workspace for real-time hand-pointing teleoperation with an Intel RealSense D435i and a Sphero BOLT.

## Requirements

- Ubuntu with ROS 1 Noetic
- Intel RealSense D435i camera
- Sphero BOLT robot
- Python 3
- Python packages: `opencv-python`, `mediapipe`, `pyrealsense2`, `numpy`

Install Python dependencies:

```bash
pip3 install opencv-python mediapipe pyrealsense2 numpy
```

## Build

```bash
cd thesis_ws
catkin_make
source devel/setup.bash
```

If custom messages are not visible, rebuild and source again:

```bash
catkin_make
source devel/setup.bash
rosmsg show hand_detection/Hand
```

## Run

```bash
roslaunch visualization thesis_system.launch
```

The launch file starts the camera, hand detection, finger analysis, workspace calibration, pointing localization, visualization, Sphero detection, and navigation nodes.

## Calibration Workflow

1. Place the 850 mm x 600 mm workspace sheet in the camera view.
2. Start the launch file.
3. If no saved calibration exists, the `Camera Calibration` window opens.
4. Click the four paper corners in order: top-left, top-right, bottom-left, bottom-right.
5. Confirm that the visualization shows the workspace and pointing target.
6. Recalibrate if the camera or workspace moves.

The calibration node saves the result to `~/.ros/paper_calibration.yaml`. On later runs, the click window may not appear because that saved calibration is loaded automatically. To force the four-click calibration again, run:

```bash
roslaunch visualization thesis_system.launch recalibrate:=true
```

You can also delete `~/.ros/paper_calibration.yaml` and launch normally. The calibration window currently uses `q` to quit; it does not use `r` to reset.

## Common Checks

Check the camera:

```bash
lsusb | grep RealSense
ls /dev/video*
```

Check ROS topics:

```bash
rostopic list
rostopic echo /pointing/target
rostopic echo /sphero/detection
```

Raw recordings should be stored locally under `bags/`; they are ignored by Git.
