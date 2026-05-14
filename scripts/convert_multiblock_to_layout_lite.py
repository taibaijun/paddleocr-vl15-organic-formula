import argparse
import json
from collections import Counter
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


def convert_split(source_sft_dir: Path, split: str, output_dir: Path, image_root: str) -> tuple[list[dict], Counter]:
    rows = read_jsonl(source_sft_dir / f"{split}.jsonl")
    out_rows = []
    distribution: Counter[int] = Counter()
    for row in rows:
        payload = lite_payload(parse_payload(row["messages"][1]["content"]))
        image_name = Path(row["images"][0]).name
        distribution[len(payload["blocks"])] += 1
        out_rows.append(
            {
                "messages": [
                    {"role": "user", "content": "<image>Chemical Structure Recognition:"},
                    {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                ],
                "images": [str(Path(image_root) / image_name)],
                "meta": {
                    "id": row.get("meta", {}).get("id"),
                    "variant": row.get("meta", {}).get("variant"),
                    "formulas": [block["formula"] for block in payload["blocks"]],
                    "layout_lite": True,
                },
            }
        )
    write_jsonl(output_dir / "sft" / f"{split}.jsonl", out_rows)
    if split == "train":
        write_jsonl(output_dir / "sft" / "smoke_train.jsonl", out_rows[:8])
    elif split == "eval":
        write_jsonl(output_dir / "sft" / "smoke_eval.jsonl", out_rows[:8])
    return out_rows, distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip multi-block SFT labels to bbox + formula + elemental_counts.")
    parser.add_argument("--source-sft-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root", required=True, help="Image directory path to write into JSONL, absolute is recommended.")
    args = parser.parse_args()

    source_sft_dir = Path(args.source_sft_dir)
    output_dir = Path(args.output_dir)
    summary = {
        "source_sft_dir": str(source_sft_dir),
        "image_root": args.image_root,
        "target": "chemical_structure blocks with bbox, formula, and elemental_counts only",
        "splits": {},
    }
    for split in ["train", "eval", "test"]:
        rows, distribution = convert_split(source_sft_dir, split, output_dir, args.image_root)
        summary["splits"][split] = {
            "pages": len(rows),
            "block_count_distribution": {str(k): distribution[k] for k in sorted(distribution)},
            "total_blocks": sum(k * distribution[k] for k in distribution),
            "file": str(output_dir / "sft" / f"{split}.jsonl"),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

