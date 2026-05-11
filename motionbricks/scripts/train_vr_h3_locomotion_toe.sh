#!/usr/bin/env bash
set -euo pipefail

cd /home/tung/GR00T-WholeBodyControl/motionbricks

PYTHON="${PYTHON:-./.venv/bin/python}"
INPUT_DIR="${INPUT_DIR:-datas/vr_h3_data/vr_h3_1}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/vr_h3_motion_features_clean_forward_walk}"
ROBOT="${ROBOT:-vr_h3}"
CLIP_CKPT="${CLIP_CKPT:-out/VR_H3-clean-forward-walk-clip.ckpt}"

ALLOW_ACTIONS_FILE="${ALLOW_ACTIONS_FILE:-scripts/vr_h3_clean_forward_walk_actions.txt}"
INCLUDE_PATTERNS="${INCLUDE_PATTERNS:-}"
EXCLUDE_PATTERNS="${EXCLUDE_PATTERNS:-}"

VQVAE_STEPS="${VQVAE_STEPS:-20000}"
POSE_STEPS="${POSE_STEPS:-50000}"
ROOT_STEPS="${ROOT_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PLOTS_DIR="${PLOTS_DIR:-out/vr_h3_clean_forward_walk_training_plots}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-20}"
PREPARE_PROGRESS_EVERY="${PREPARE_PROGRESS_EVERY:-1000}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

echo "== VR H3 locomotion training with dummy toes =="
echo "Python: ${PYTHON}"
echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Robot: ${ROBOT}"
echo "Allow actions file: ${ALLOW_ACTIONS_FILE}"
echo "Include patterns: ${INCLUDE_PATTERNS}"
echo "Exclude patterns: ${EXCLUDE_PATTERNS}"
echo "Steps: vqvae=${VQVAE_STEPS}, pose=${POSE_STEPS}, root=${ROOT_STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Num workers: ${NUM_WORKERS}"
echo "Plots dir: ${PLOTS_DIR}"
echo "Clip ckpt: ${CLIP_CKPT}"
echo "Prepare progress every: ${PREPARE_PROGRESS_EVERY}"
echo "Skip prepare: ${SKIP_PREPARE}"

DATASET_PT="${OUTPUT_DIR}/motions.pt"

if [[ "${SKIP_PREPARE}" == "1" ]]; then
  if [[ ! -f "${DATASET_PT}" ]]; then
    echo "SKIP_PREPARE=1 but dataset does not exist: ${DATASET_PT}" >&2
    exit 1
  fi
  echo
  echo "== Skipping dataset prepare =="
  echo "Using existing dataset: ${DATASET_PT}"
else
  echo
  echo "== Preparing locomotion dataset =="
  "${PYTHON}" scripts/prepare_vr_h3_data.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --allow_actions_file "${ALLOW_ACTIONS_FILE}" \
    --include_patterns "${INCLUDE_PATTERNS}" \
    --exclude_patterns "${EXCLUDE_PATTERNS}" \
    --progress_every "${PREPARE_PROGRESS_EVERY}"
fi

echo
echo "== Training VQVAE =="
"${PYTHON}" scripts/train_vqvae.py \
  --robot "${ROBOT}" \
  --dataset_pt "${DATASET_PT}" \
  --max_steps "${VQVAE_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

echo
echo "== Training pose model =="
"${PYTHON}" scripts/train_pose.py \
  --robot "${ROBOT}" \
  --dataset_pt "${DATASET_PT}" \
  --max_steps "${POSE_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

echo
echo "== Training root model =="
"${PYTHON}" scripts/train_root.py \
  --robot "${ROBOT}" \
  --dataset_pt "${DATASET_PT}" \
  --max_steps "${ROOT_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

echo
echo "== Visualizing training metrics =="
"${PYTHON}" scripts/visualize_training_metrics.py \
  --output_dir "${PLOTS_DIR}" \
  --smooth_window "${SMOOTH_WINDOW}"

echo
echo "== Building demo clip cache =="
"${PYTHON}" scripts/build_vr_h3_clip_cache.py \
  --dataset_pt "${DATASET_PT}" \
  --output "${CLIP_CKPT}"

echo
echo "== Done =="
echo "Dataset: ${DATASET_PT}"
echo "Clip ckpt: ${CLIP_CKPT}"
echo "VQVAE metrics: out/motionbricks_vr_h3_vqvae/version_1/training_logs/latest/metrics.csv"
echo "Pose metrics: out/motionbricks_vr_h3_pose/version_1/training_logs/latest/metrics.csv"
echo "Root metrics: out/motionbricks_vr_h3_root/version_1/training_logs/latest/metrics.csv"
echo "Metric plots: ${PLOTS_DIR}"
echo "Metric summary: ${PLOTS_DIR}/summary.md"
