#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/百度ocr比赛

CUDA_VISIBLE_DEVICES=0 \
NNODES=1 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=36704 \
/home/ubuntu/miniforge3/bin/conda run -n cudabase \
  paddleformers-cli train \
  /mnt/d/百度ocr比赛/train_configs/paddleocr_vl15_formula_organic_ability_v6_mixed_lite_lora_r16_b2ga8_1epoch.yaml

