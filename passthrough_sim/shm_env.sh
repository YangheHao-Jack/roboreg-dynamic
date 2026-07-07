# Source this in EVERY shell that touches the pixel topics:
#   the producer shell (bridge sidecar inherits), the ros2 launch shell
#   (rectifier inherits), and the consumer shell.
#
#   source ~/passthrough_sim/shm_env.sh
#
# A participant missing this profile still discovers over UDP, but its
# image_rect hop falls back to fragmented UDP loopback (~20+ ms wire).
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/passthrough_sim/fastdds_shm.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
