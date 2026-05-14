#!/usr/bin/env bash
set -euo pipefail

cd .
source ${VENV_PATH}/bin/activate

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
NNODES=1 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT="${MASTER_PORT:-36704}" \
paddleformers-cli train \
  ./configs/paddleocr_vl15_formula_organic_ability_v6_mixed_lite_lora_r16_b2ga8_1epoch_server.yaml

