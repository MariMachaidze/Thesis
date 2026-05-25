# Gesture-Based Teleoperation with Hand Pointing and Sphero BOLT

This repository contains the ROS 1 system and curated evaluation results for a thesis project on gesture-based teleoperation. An Intel RealSense D435i camera detects a user's pointing hand, estimates the target location on a calibrated tabletop workspace, and uses that target to guide a Sphero BOLT robot in closed loop.

## System Overview

The pipeline combines real-time perception, geometric pointing localization, and robot navigation:

```text
D435i RGB + depth frames
  -> MediaPipe hand landmark detection
  -> 3D finger reconstruction and straightness gating
  -> workspace calibration and ray-plane intersection
  -> pointing target in workspace coordinates
  -> Sphero detection and closed-loop navigation
```

The main ROS workspace is [`thesis_ws/`](thesis_ws/). Source packages live in [`thesis_ws/src/`](thesis_ws/src/) and include camera streaming, hand detection, finger analysis, workspace calibration, pointing localization, visualization, Sphero detection, and Sphero navigation.

## Quick Start

Prerequisites:

- Ubuntu with ROS 1 Noetic
- Intel RealSense D435i
- Sphero BOLT
- Python 3 packages: `opencv-python`, `mediapipe`, `pyrealsense2`, `numpy`

Build and source the workspace:

```bash
cd thesis_ws
catkin_make
source devel/setup.bash
```

Launch the full system:

```bash
roslaunch visualization thesis_system.launch
```

On first run, calibrate the paper workspace by clicking the four visible corners when the calibration window opens. If the click window does not appear, the node probably loaded a saved calibration from `~/.ros/paper_calibration.yaml`; force a new calibration with:

```bash
roslaunch visualization thesis_system.launch recalibrate:=true
```

After calibration, point at a location on the workspace; the estimated target is published and the Sphero navigation node can use it as a goal.

## Repository Structure

```text
thesis_ws/      ROS workspace and source packages
docs/           setup, architecture, methodology, and result notes
results/        curated thesis result summaries, plots, CSVs, and analysis scripts
data/           placeholder for local raw data placement
```

Useful entry points:

- [Setup guide](docs/setup.md)
- [System architecture](docs/architecture.md)
- [Methodology](docs/methodology.md)
- [Results overview](docs/results.md)
- [Curated results](results/README.md)

## Results Summary

Curated results are included under [`results/`](results/) so the main thesis findings can be inspected without downloading raw ROS bags.

| Section | Focus | Key result |
| --- | --- | --- |
| [5.1](results/5_1_pointing_localization/) | Pointing localization accuracy | Mean localization error: 63.9 mm across 133 curated trials |
| [5.2](results/5_2_robot_navigation/) | Robot navigation accuracy | Mean navigation error: 3.10 cm; 77.8% within 5 cm |
| [5.3](results/5_3_pointing_latency/) | Pointing pipeline latency | Mean latency: 55.44 ms; 98.1% of samples <= 100 ms |
| [5.4](results/5_4_navigation_latency/) | Navigation latency | Navigation time distributions and distance/latency analysis |
| [5.5](results/5_5_navigation_success/) | Navigation success | Success rate by region and workspace-level success plots |

The raw `.bag` recordings are intentionally excluded from Git because they are large. Place raw recordings locally under `bags/` when reproducing analyses; see [`data/README.md`](data/README.md).

## Tuning Note

Performance depends on the workspace geometry, lighting, camera placement, and Sphero response. Tuning the One Euro filter parameters, straightness thresholds, target commit logic, and controller gains can reduce jitter, improve stability, and tighten final accuracy/latency for a new setup.

## Documentation

The concise public documentation is in [`docs/`](docs/). The workspace README in [`thesis_ws/README.md`](thesis_ws/README.md) remains as a lower-level ROS reference.
