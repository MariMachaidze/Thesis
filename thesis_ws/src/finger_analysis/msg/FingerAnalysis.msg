Header header
geometry_msgs/Point knuckle_2d
geometry_msgs/Point tip_2d
geometry_msgs/Point knuckle_3d
geometry_msgs/Point tip_3d
geometry_msgs/Vector3 direction_3d
float32 straightness_raw        # cosine score before any filtering
float32 straightness_one_euro   # after One Euro Filter
float32 straightness_score      # One Euro filtered score (authoritative output)
bool is_straight
float32 confidence
