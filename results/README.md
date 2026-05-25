# Curated Results

This directory contains the thesis result artifacts that are suitable for GitHub: written summaries, CSV files, plots, and the analysis scripts used to generate the figures.

Raw ROS bags are intentionally excluded from the repository because they are large. To rerun analyses from raw recordings, place local data under `bags/` or another external archive and update the analysis script paths as needed.

## Sections

| Directory | Description |
| --- | --- |
| `5_1_pointing_localization/` | Pointing target localization accuracy across targets and hand heights |
| `5_2_robot_navigation/` | Sphero navigation accuracy and trajectory behavior |
| `5_3_pointing_latency/` | Pointing localization latency and real-time responsiveness |
| `5_4_navigation_latency/` | Robot navigation latency and distance relationships |
| `5_5_navigation_success/` | Success rate across the workspace and by region |

Each section keeps its local `analysis.py` next to the outputs it generated.
