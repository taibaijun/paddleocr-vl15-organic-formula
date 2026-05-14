#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/百度ocr比赛
OUTPUT_DIR=/mnt/d/百度ocr比赛/output/paddleocr_vl15_formula_curriculum_v2_counts_lora_r16_b2ga8_2epoch
case "$OUTPUT_DIR" in
  /mnt/d/百度ocr比赛/output/paddleocr_vl15_formula_curriculum_v2_counts_lora_r16_b2ga8_2epoch) rm -rf "$OUTPUT_DIR" ;;
  *) echo "Refusing to remove unexpected output dir: $OUTPUT_DIR" >&2; exit 1 ;;
esac

CUDA_VISIBLE_DEVICES=0 \
NNODES=1 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=36695 \
/home/ubuntu/miniforge3/bin/conda run -n cudabase \
  paddleformers-cli train \
  /mnt/d/百度ocr比赛/train_configs/paddleocr_vl15_formula_curriculum_v2_counts_lora_r16_b2ga8_2epoch.yaml

