#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/百度ocr比赛
source /home/ubuntu/miniforge3/bin/activate cudabase

for target in output/organic_dataset_5000 output/organic_paddleformers_sft_5000; do
  abs=$(realpath -m "$target")
  case "$abs" in
    /mnt/d/百度ocr比赛/output/organic_dataset_5000|/mnt/d/百度ocr比赛/output/organic_paddleformers_sft_5000)
      rm -rf "$abs"
      ;;
    *)
      echo "refuse to remove $abs"
      exit 9
      ;;
  esac
done

python generate_organic_dataset_augmented.py \
  --output-dir output/organic_dataset_5000 \
  --count 5000 \
  --variants both \
  --progress-every 250

python convert_organic_to_paddleformers_sft.py \
  --labels output/organic_dataset_5000/labels.jsonl \
  --output-dir output/organic_paddleformers_sft_5000 \
  --target-format json \
  --variants both \
  --copy-images \
  --eval-ratio 0.1 \
  --smoke-train-size 8 \
  --smoke-eval-size 4

echo DONE

