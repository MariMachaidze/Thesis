# Methodology

The evaluation measures a complete perception-to-action chain: a user points at a location on a calibrated tabletop workspace, the system estimates the intended target, and the Sphero BOLT navigates toward it.

## Pointing Localization

The pointing pipeline detects 21 MediaPipe hand landmarks and uses the index finger joints to estimate a 3D pointing ray. Landmark pixels are paired with aligned depth values from the D435i and back-projected using the pinhole camera model.

The index finger is accepted as a pointing gesture only when it is sufficiently straight. The system smooths 3D joints and target coordinates to reduce depth jitter, then intersects the pointing ray with the calibrated workspace plane. The final target is expressed as normalized `(u, v)` coordinates and metric workspace coordinates.

## Workspace Calibration

The workspace is an 850 mm x 600 mm planar sheet. Calibration uses four manually clicked corners: top-left, top-right, bottom-left, and bottom-right. Each corner is back-projected into 3D, a plane is fit from the reconstructed points, and the top-left corner defines the workspace origin.

The calibrated axes allow both pointing targets and Sphero detections to be represented in one common 2D coordinate system.

## Robot Navigation

The Sphero is detected in the workspace view and controlled in closed loop. The navigation node receives a pointing target, compares it with the current robot position, and commands heading/speed updates until the robot reaches the target region.

## Evaluation Metrics

Pointing localization is evaluated with Euclidean error between the estimated point and known target coordinates. Latency is measured from the generated pointing samples. Robot navigation is evaluated with minimum distance to the intended target, success rate within a fixed tolerance, and navigation time.

The curated result folders contain the scripts, CSVs, plots, and written summaries used for the thesis result sections.
