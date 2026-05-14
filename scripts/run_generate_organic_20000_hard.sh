#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/百度ocr比赛
source /home/ubuntu/miniforge3/bin/activate cudabase

for target in output/organic_dataset_20000_hard output/organic_paddleformers_sft_20000_hard; do
  abs=$(realpath -m "$target")
  case "$abs" in
    /mnt/d/百度ocr比赛/output/organic_dataset_20000_hard|/mnt/d/百度ocr比赛/output/organic_paddleformers_sft_20000_hard)
      rm -rf "$abs"
      ;;
    *)
      echo "refuse to remove $abs"
      exit 9
      ;;
  esac
done

python generate_organic_dataset_augmented.py \
  --output-dir output/organic_dataset_20000_hard \
  --count 16000 \
  --variants both \
  --progress-every 500

python scripts/build_hard_organic_samples.py \
  --labels output/organic_dataset_20000_hard/labels.jsonl \
  --output-dir output/organic_dataset_20000_hard \
  --count 4000 \
  --progress-every 250

python convert_organic_to_paddleformers_sft.py \
  --labels output/organic_dataset_20000_hard/labels.jsonl \
  --output-dir output/organic_paddleformers_sft_20000_hard \
  --target-format json \
  --variants both \
  --eval-ratio 0.05 \
  --smoke-train-size 16 \
  --smoke-eval-size 8

echo DONE

