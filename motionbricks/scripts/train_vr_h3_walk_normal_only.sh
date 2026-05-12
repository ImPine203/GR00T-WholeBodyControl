#!/usr/bin/env bash
set -euo pipefail

cd /home/tung/GR00T-WholeBodyControl/motionbricks

ALLOW_ACTIONS_FILE="${ALLOW_ACTIONS_FILE:-scripts/vr_h3_walk_normal_only_actions.txt}" \
OUTPUT_DIR="${OUTPUT_DIR:-datasets/vr_h3_motion_features_walk_normal_only}" \
CLIP_CKPT="${CLIP_CKPT:-out/VR_H3-walk-normal-only-clip.ckpt}" \
PLOTS_DIR="${PLOTS_DIR:-out/vr_h3_walk_normal_only_training_plots}" \
bash scripts/train_vr_h3_locomotion_toe.sh
