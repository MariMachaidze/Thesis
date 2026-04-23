Header header
geometry_msgs/Point knuckle_2d
geometry_msgs/Point tip_2d
geometry_msgs/Point knuckle_3d
geometry_msgs/Point tip_3d
geometry_msgs/Vector3 direction_3d
float32 straightness_raw        # cosine score before any filtering
float32 straightness_ema        # after median-3 buffer + EMA
float32 straightness_one_euro   # after One Euro Filter (0 if disabled)
float32 straightness_score      # active output: EMA or One Euro depending on ~use_one_euro
bool is_straight
float32 confidence
