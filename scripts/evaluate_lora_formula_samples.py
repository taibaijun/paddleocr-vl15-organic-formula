import argparse
import json
from collections import Counter
from pathlib import Path

import paddle
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor

from run_paddleocr_vl_paddle_infer import PROMPTS, patch_processor_call, run_one


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_json_text(text: str) -> dict | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def formulas_from_payload(payload: dict | None) -> list[str]:
    if not payload:
        return []
    formulas = []
    for block in payload.get("blocks", []):
        formula = block.get("formula")
        if isinstance(formula, str) and formula.strip():
            formulas.append(formula.strip())
    return formulas


def multiset_overlap(left: list[str], right: list[str]) -> int:
    left_counts = Counter(left)
    right_counts = Counter(right)
    return sum((left_counts & right_counts).values())


def counts_from_payload(payload: dict | None) -> list[dict[str, int]]:
    if not payload:
        return []
    counts = []
    for block in payload.get("blocks", []):
        value = block.get("elemental_counts")
        if isinstance(value, dict):
            clean = {}
            for key, count in value.items():
                try:
                    clean[str(key)] = int(count)
                except (TypeError, ValueError):
                    clean[str(key)] = count
            counts.append(clean)
    return counts


def resolve_image_path(dataset_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return dataset_dir / path


def infer_prompt(row: dict) -> str:
    user_content = row["messages"][0]["content"]
    if "OCR:" in user_content:
        return PROMPTS["ocr"]
    return PROMPTS["chemical"]


def build_summary(
    rows_seen: int,
    exact: int,
    set_match: int,
    counts_exact: int,
    counts_seen: int,
    block_count_match: int,
    formula_hits: int,
    formula_targets: int,
    formula_predictions: int,
    results: list[dict],
) -> dict:
    return {
        "total": rows_seen,
        "exact_match": exact,
        "exact_accuracy": exact / rows_seen if rows_seen else 0.0,
        "set_match": set_match,
        "set_accuracy": set_match / rows_seen if rows_seen else 0.0,
        "counts_exact": counts_exact,
        "counts_total": counts_seen,
        "counts_accuracy": counts_exact / counts_seen if counts_seen else None,
        "block_count_match": block_count_match,
        "block_count_accuracy": block_count_match / rows_seen if rows_seen else 0.0,
        "formula_item_hits": formula_hits,
        "formula_item_targets": formula_targets,
        "formula_item_predictions": formula_predictions,
        "formula_item_recall": formula_hits / formula_targets if formula_targets else 0.0,
        "formula_item_precision": formula_hits / formula_predictions if formula_predictions else 0.0,
        "formula_item_f1": (2 * formula_hits / (formula_targets + formula_predictions))
        if formula_targets + formula_predictions
        else 0.0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate formula extraction on a small PaddleOCR-VL LoRA sample set.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--official-template", action="store_true")
    parser.add_argument("--save-every", type=int, default=0, help="Write a partial JSON result every N samples.")
    args = parser.parse_args()

    paddle.set_device("gpu:0")
    model = AutoModelForConditionalGeneration.from_pretrained(
        args.model_path,
        convert_from_hf=True,
    ).eval()
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"

    from paddleformers.peft import LoRAModel

    lora_model = LoRAModel.from_pretrained(model, args.lora_path)
    model = lora_model.model.eval()

    processor = AutoProcessor.from_pretrained(args.lora_path)
    patch_processor_call(processor)

    dataset_dir = Path(args.dataset_dir)
    rows = read_jsonl(Path(args.jsonl))
    if args.min_id is not None:
        rows = [row for row in rows if int(row.get("meta", {}).get("id", "0").split("_", 1)[0]) >= args.min_id]
    rows = rows[: args.limit]
    results = []
    exact = 0
    set_match = 0
    counts_exact = 0
    counts_seen = 0
    block_count_match = 0
    formula_hits = 0
    formula_targets = 0
    formula_predictions = 0
    for row in rows:
        target_payload = parse_json_text(row["messages"][1]["content"])
        target_formulas = formulas_from_payload(target_payload)
        target_counts = counts_from_payload(target_payload)
        image_path = resolve_image_path(dataset_dir, row["images"][0])
        image = Image.open(image_path).convert("RGB")
        pred_text = run_one(
            model=model,
            processor=processor,
            image=image,
            prompt=infer_prompt(row),
            max_new_tokens=args.max_new_tokens,
            official_template=args.official_template,
        )
        pred_payload = parse_json_text(pred_text)
        pred_formulas = formulas_from_payload(pred_payload)
        pred_counts = counts_from_payload(pred_payload)
        exact_ok = pred_formulas == target_formulas
        set_ok = sorted(pred_formulas) == sorted(target_formulas)
        counts_ok = pred_counts == target_counts if target_counts else None
        formula_overlap = multiset_overlap(target_formulas, pred_formulas)
        exact += int(exact_ok)
        set_match += int(set_ok)
        block_count_match += int(len(pred_formulas) == len(target_formulas))
        formula_hits += formula_overlap
        formula_targets += len(target_formulas)
        formula_predictions += len(pred_formulas)
        if target_counts:
            counts_seen += 1
            counts_exact += int(bool(counts_ok))
        results.append(
            {
                "id": row.get("meta", {}).get("id"),
                "variant": row.get("meta", {}).get("variant"),
                "target_formulas": target_formulas,
                "pred_formulas": pred_formulas,
                "target_counts": target_counts,
                "pred_counts": pred_counts,
                "exact_match": exact_ok,
                "set_match": set_ok,
                "counts_match": counts_ok,
                "formula_overlap": formula_overlap,
                "target_block_count": len(target_formulas),
                "pred_block_count": len(pred_formulas),
                "prediction_escaped": pred_text.encode("unicode_escape").decode("ascii"),
            }
        )
        print(
            f"{len(results):03d} {row.get('meta', {}).get('id')} "
            f"exact={exact_ok} counts={counts_ok} set={set_ok} target={target_formulas} pred={pred_formulas}"
            ,
            flush=True,
        )
        if args.save_every and len(results) % args.save_every == 0:
            partial = build_summary(
                len(results),
                exact,
                set_match,
                counts_exact,
                counts_seen,
                block_count_match,
                formula_hits,
                formula_targets,
                formula_predictions,
                results,
            )
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(partial, ensure_ascii=True, indent=2), encoding="utf-8")

    summary = build_summary(
        len(rows),
        exact,
        set_match,
        counts_exact,
        counts_seen,
        block_count_match,
        formula_hits,
        formula_targets,
        formula_predictions,
        results,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
