# Results

Curated thesis outputs are stored in [`../results/`](../results/). These folders are small enough for GitHub and include summary text, plots, CSV files, and analysis scripts.

## Included Result Sets

| Folder | Thesis section | Contents |
| --- | --- | --- |
| `results/5_1_pointing_localization/` | 5.1 | Pointing accuracy summaries, CSVs, and localization plots |
| `results/5_2_robot_navigation/` | 5.2 | Navigation accuracy and trajectory analysis |
| `results/5_3_pointing_latency/` | 5.3 | Per-sample pointing latency analysis |
| `results/5_4_navigation_latency/` | 5.4 | Navigation time distributions and distance relationships |
| `results/5_5_navigation_success/` | 5.5 | Success-rate analysis across the workspace |

## Headline Findings

- Pointing localization achieved a mean curated error of 63.9 mm across 133 trial-level estimates.
- Robot navigation reached a mean minimum distance of 3.10 cm over 275 arrived trials.
- Pointing updates averaged 55.44 ms, with 98.1% of samples below the 100 ms interaction budget.
- Navigation success at the 5 cm threshold was 77.8%, increasing substantially under a broader 8 cm tolerance.

## Raw Data Policy

Raw ROS bags and depth captures are not committed because they are very large. They should be stored locally under `bags/` or archived externally if full replay is needed. The committed result folders preserve the public evidence needed to inspect the thesis findings without requiring the raw recordings.
