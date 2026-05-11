#!/usr/bin/env bash
set -euo pipefail

cd /home/tung/GR00T-WholeBodyControl/motionbricks

PYTHON="${PYTHON:-./.venv/bin/python}"
INPUT_DIR="${INPUT_DIR:-datas/vr_h3_data/vr_h3_1}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/vr_h3_motion_features_full_all}"
ROBOT="${ROBOT:-vr_h3}"

VQVAE_STEPS="${VQVAE_STEPS:-20000}"
POSE_STEPS="${POSE_STEPS:-50000}"
ROOT_STEPS="${ROOT_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PLOTS_DIR="${PLOTS_DIR:-out/vr_h3_training_plots}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-20}"
PREPARE_PROGRESS_EVERY="${PREPARE_PROGRESS_EVERY:-1000}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

echo "== VR H3 full-data training =="
echo "Python: ${PYTHON}"
echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Robot: ${ROBOT}"
echo "Steps: vqvae=${VQVAE_STEPS}, pose=${POSE_STEPS}, root=${ROOT_STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Num workers: ${NUM_WORKERS}"
echo "Plots dir: ${PLOTS_DIR}"
echo "Prepare progress every: ${PREPARE_PROGRESS_EVERY}"

echo
echo "== Preparing full dataset =="
"${PYTHON}" scripts/prepare_vr_h3_data.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --progress_every "${PREPARE_PROGRESS_EVERY}"

DATASET_PT="${OUTPUT_DIR}/motions.pt"

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
echo "== Done =="
echo "Dataset: ${DATASET_PT}"
echo "VQVAE metrics: out/motionbricks_vr_h3_vqvae/version_1/training_logs/latest/metrics.csv"
echo "Pose metrics: out/motionbricks_vr_h3_pose/version_1/training_logs/latest/metrics.csv"
echo "Root metrics: out/motionbricks_vr_h3_root/version_1/training_logs/latest/metrics.csv"
echo "Metric plots: ${PLOTS_DIR}"
echo "Metric summary: ${PLOTS_DIR}/summary.md"
