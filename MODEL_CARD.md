# Model Card: PaddleOCR-VL-1.5 Organic Formula LoRA

## Overview

This repository provides a LoRA adapter for recognizing organic chemical structure images with PaddleOCR-VL-1.5. The model outputs structured JSON containing molecular formulas, elemental counts, and page-level structure blocks.

## Base Model

The adapter requires a separately downloaded PaddleOCR-VL-1.5 base model.

```text
/path/to/PaddleOCR-VL-1.5
```

## Adapter

The LoRA adapter is stored in:

```text
adapter/
```

It is not a standalone model. Load it together with the PaddleOCR-VL-1.5 base model.

## Output Schema

```json
{
  "blocks": [
    {
      "formula": "C16H32O2",
      "elemental_counts": {
        "C": 16,
        "H": 32,
        "O": 2
      }
    }
  ]
}
```

## Evaluation Snapshot

Regular curated set `regular_curated_90`:

| Metric | Value |
|---|---:|
| Samples | 90 |
| Exact accuracy | 78.89% |
| Set accuracy | 78.89% |
| Counts accuracy | 78.89% |
| Block count accuracy | 100.00% |
| Formula item F1 | 88.24% |

Hard multi-structure probe set `hard_multistructure_probe_30`:

| Metric | Value |
|---|---:|
| Samples | 30 |
| Exact accuracy | 13.33% |
| Set accuracy | 13.33% |
| Counts accuracy | 13.33% |
| Block count accuracy | 100.00% |
| Formula item F1 | 67.50% |

Combined evaluation `curated_combined_120`:

| Metric | Value |
|---|---:|
| Samples | 120 |
| Exact accuracy | 62.50% |
| Set accuracy | 62.50% |
| Counts accuracy | 62.50% |
| Block count accuracy | 100.00% |
| Formula item F1 | 79.66% |

Result files:

```text
eval/regular_curated_90.json
eval/hard_multistructure_probe_30.json
eval/curated_combined_120.json
```

## Intended Use

- Organic chemical structure OCR
- Chemistry worksheet and textbook image parsing
- Experiment-note and paper-figure information extraction
- PaddleOCR-VL LoRA fine-tuning experiments
