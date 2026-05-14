import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_payload(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("assistant message does not contain a JSON object")
    return json.loads(text[start : end + 1])


def lite_payload(payload: dict) -> dict:
    blocks = []
    for block in payload.get("blocks", []):
        blocks.append(
            {
                "type": "chemical_structure",
                "bbox": block["bbox"],
                "formula": block["formula"],
                "elemental_counts": block.get("elemental_counts", {}),
            }
        )
    return {"blocks": blocks}


def make_single_lite_rows(rows: list[dict], image_root: str) -> list[dict]:
    out = []
    for row in rows:
        payload = lite_payload(parse_payload(row["messages"][1]["content"]))
        image_name = Path(row["images"][0]).name
        out.append(
            {
                "messages": [
                    {"role": "user", "content": "<image>Chemical Structure Recognition:"},
                    {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                ],
                "images": [str(Path(image_root) / image_name)],
                "meta": {
                    "id": row.get("meta", {}).get("id"),
                    "variant": "single_lite_rehearsal",
                    "formulas": [block["formula"] for block in payload["blocks"]],
                    "layout_lite": True,
                },
            }
        )
    return out


def sample_rows(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    if count < 0:
        return rows
    if count > len(rows):
        raise ValueError(f"requested {count} rows, only found {len(rows)}")
    rows = rows[:]
    rng.shuffle(rows)
    return rows[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced lite SFT dataset from multiblock lite pages plus single-page rehearsal.")
    parser.add_argument("--multi-lite-sft-dir", required=True)
    parser.add_argument("--single-sft-dir", required=True)
    parser.add_argument("--single-image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--single-train-count", type=int, default=12000)
    parser.add_argument("--multi-eval-count", type=int, default=500)
    parser.add_argument("--single-eval-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026050925)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    multi_dir = Path(args.multi_lite_sft_dir)
    single_dir = Path(args.single_sft_dir)
    output_dir = Path(args.output_dir)

    train_multi = read_jsonl(multi_dir / "train.jsonl")
    train_single = make_single_lite_rows(
        sample_rows(read_jsonl(single_dir / "train.jsonl"), args.single_train_count, rng),
        args.single_image_root,
    )
    train_rows = train_multi + train_single
    rng.shuffle(train_rows)

    eval_multi = sample_rows(read_jsonl(multi_dir / "eval.jsonl"), args.multi_eval_count, rng)
    eval_single = make_single_lite_rows(
        sample_rows(read_jsonl(single_dir / "eval.jsonl"), args.single_eval_count, rng),
        args.single_image_root,
    )
    eval_rows = eval_multi + eval_single
    rng.shuffle(eval_rows)

    # Keep the original split-specific test rows available for quick smoke checks; final reporting is done separately.
    test_multi = sample_rows(read_jsonl(multi_dir / "test.jsonl"), 200, rng)
    test_single = make_single_lite_rows(sample_rows(read_jsonl(single_dir / "test.jsonl"), 200, rng), args.single_image_root)
    test_rows = test_multi + test_single
    rng.shuffle(test_rows)

    write_jsonl(output_dir / "sft" / "train.jsonl", train_rows)
    write_jsonl(output_dir / "sft" / "eval.jsonl", eval_rows)
    write_jsonl(output_dir / "sft" / "test.jsonl", test_rows)
    write_jsonl(output_dir / "sft" / "smoke_train.jsonl", train_rows[:8])
    write_jsonl(output_dir / "sft" / "smoke_eval.jsonl", eval_rows[:8])

    summary = {
        "seed": args.seed,
        "target": "balanced lite: multiblock pages plus single-page rehearsal",
        "train": {
            "multi_pages": len(train_multi),
            "single_pages": len(train_single),
            "total": len(train_rows),
        },
        "eval": {
            "multi_pages": len(eval_multi),
            "single_pages": len(eval_single),
            "total": len(eval_rows),
        },
        "test": {
            "multi_pages": len(test_multi),
            "single_pages": len(test_single),
            "total": len(test_rows),
        },
        "files": {
            "train": str(output_dir / "sft" / "train.jsonl"),
            "eval": str(output_dir / "sft" / "eval.jsonl"),
            "test": str(output_dir / "sft" / "test.jsonl"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

