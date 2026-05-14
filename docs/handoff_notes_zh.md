# PaddleOCR-VL 有机化学结构识别项目交接说明

## 1. 当前交付状态

当前训练任务已经停止，仓库中保留了最佳 LoRA adapter、训练配置、评测脚本、当前评测结果、测试样本和展示文档。基座模型未包含在仓库中，推理时需要单独准备 PaddleOCR-VL-1.5。

最佳 LoRA：

```text
adapter/
```

基座模型路径示例：

```text
/path/to/PaddleOCR-VL-1.5
```

## 2. 训练方法

本项目使用 PaddleFormers 官方 PaddleOCR-VL 微调流程，训练方式为 LoRA SFT。基座模型参数保持冻结，训练产物为独立 LoRA adapter。推理时需要同时加载 PaddleOCR-VL-1.5 基座模型和 `adapter/`。

训练样本格式为 image-text SFT messages，每条样本包含：

| 部分 | 内容 |
|---|---|
| image | 化学结构图或多结构页面 |
| instruction | 要求模型识别化学结构并输出 JSON |
| answer | 标准 JSON，包含 `blocks`、`formula`、`elemental_counts` |

主要训练策略：

| 策略 | 用途 |
|---|---|
| LoRA rank 16 | 控制训练参数量，降低显存压力 |
| bf16 | 提升训练吞吐并控制显存 |
| batch size + gradient accumulation | 在显存允许范围内扩大等效 batch |
| sharding stage1 | 多卡训练时降低优化器状态显存 |
| formula/layout 混合任务 | 同时学习分子式识别和页面结构化输出 |
| hard replay | 强化复杂页面样式 |
| eval/test 固定集 | 多轮实验之间保持可比性 |

关键训练配置在：

```text
configs/
```

最佳版本配置：

```text
configs/paddleocr_vl15_formula_organic_ability_v6_hard_formula_lora_r16_b3ga2_round7_lr3e7.yaml
```

## 3. 输出格式

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

## 4. 当前评测结果

当前开源仓库采用两个评测集分开记录指标：

| 评测集 | 样本数 | exact accuracy | set accuracy | counts accuracy | block count accuracy | formula F1 | 结果文件 |
|---|---:|---:|---:|---:|---:|---:|---|
| `regular_curated_90` | 90 | 0.7889 | 0.7889 | 0.7889 | 1.0000 | 0.8824 | `eval/regular_curated_90.json` |
| `hard_multistructure_probe_30` | 30 | 0.1333 | 0.1333 | 0.1333 | 1.0000 | 0.6750 | `eval/hard_multistructure_probe_30.json` |
| `curated_combined_120` | 120 | 0.6250 | 0.6250 | 0.6250 | 1.0000 | 0.7966 | `eval/curated_combined_120.json` |

仓库同时包含 100 条本地验证样本：

```text
test_data/test_100.jsonl
test_data/images/
eval/test_100_round7_eval.json
```

## 5. 关键目录

```text
adapter/          # LoRA adapter
configs/          # PaddleFormers training configs
scripts/          # inference, evaluation, and training helper scripts
eval/             # current evaluation results
examples/         # visual demo images
test_data/        # 100 local verification samples
docs/             # technical documents
```

## 6. 推理命令

```bash
python scripts/run_paddleocr_vl_paddle_infer.py \
  --model-path /path/to/PaddleOCR-VL-1.5 \
  --lora-path adapter \
  --image examples/demo_four_structures.png \
  --output result.json \
  --tasks chemical \
  --max-new-tokens 768
```

## 7. 评测命令

```bash
python scripts/evaluate_lora_formula_samples.py \
  --model-path /path/to/PaddleOCR-VL-1.5 \
  --lora-path adapter \
  --jsonl test_data/test_100.jsonl \
  --dataset-dir test_data \
  --output eval_test_100.json \
  --limit 100 \
  --max-new-tokens 768
```
