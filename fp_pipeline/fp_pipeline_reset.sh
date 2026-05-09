#!/bin/bash
# fp_pipeline_reset.sh
#
# One-shot reset for the FP+IPCAI pipeline. Run before every launch
# so there's no leftover state from a previous run.
#
# Usage: bash ~/fp_pipeline/fp_pipeline_reset.sh

set -u

echo "[reset] killing pipeline processes..."
pkill -9 -f stereo_pipeline_live      2>/dev/null
pkill -9 -f stereo_pipeline_live_handoff 2>/dev/null
pkill -9 -f fp_pose_recorder          2>/dev/null
pkill -9 -f qpf2_receiver             2>/dev/null
pkill -9 -f bag_to_qpf2               2>/dev/null
pkill -9 -f bag_joint_replayer        2>/dev/null
pkill -9 -f bag_first_joint_publisher 2>/dev/null
pkill -9 -f bag_first_caminfo_publisher 2>/dev/null
pkill -9 -f stereo_depth_saver        2>/dev/null
pkill -9 -f bake_node                 2>/dev/null
pkill -9 -f component_container_mt    2>/dev/null
sleep 1

echo "[reset] removing runtime files..."
rm -rf /tmp/fp_bake_runtime
rm -rf /tmp/fp_init_pids
rm -f  /tmp/qpf2_start
rm -rf "${HOME}/fp_init_debug"

# Optional: clear any DDS shared-memory leftovers from prior runs.
# Cyclone DDS stores nothing on disk by default; Fast-DDS uses /dev/shm.
echo "[reset] clearing DDS shared memory..."
rm -rf /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null

# Kill the rclpy daemon if present (rare, but it can hold latched msgs).
ros2 daemon stop 2>/dev/null

echo "[reset] done."
