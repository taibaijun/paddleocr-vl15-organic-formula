#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/百度ocr比赛

output_dir=output/paddleocr_vl15_caption_plain_lora_safe50
abs_output=$(realpath -m "$output_dir")
case "$abs_output" in
  /mnt/d/百度ocr比赛/output/paddleocr_vl15_caption_plain_lora_safe50)
    rm -rf "$abs_output"
    ;;
  *)
    echo "refuse to remove $abs_output"
    exit 9
    ;;
esac

CUDA_VISIBLE_DEVICES=0 \
NNODES=1 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=36681 \
/home/ubuntu/miniforge3/bin/conda run -n cudabase \
  paddleformers-cli train \
  /mnt/d/百度ocr比赛/train_configs/paddleocr_vl15_caption_plain_lora_safe50.yaml

