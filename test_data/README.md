# Test Data

This directory contains 100 sample test images and labels from the organic formula recognition evaluation split.

```text
test_100.jsonl
images/
```

Each JSONL row follows the PaddleOCR-VL messages format. Image paths are relative to this directory, so evaluation can use:

The included `round7_hard_formula` evaluation result is available at:

```text
../eval/test_100_round7_eval.json
```

```bash
python scripts/evaluate_lora_formula_samples.py \
  --model-path /path/to/PaddleOCR-VL-1.5 \
  --lora-path adapter \
  --jsonl test_data/test_100.jsonl \
  --dataset-dir test_data \
  --output eval_result.json \
  --limit 100 \
  --max-new-tokens 768
```
