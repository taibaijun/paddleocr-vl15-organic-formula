import argparse
import json
import random
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


def sample_rows(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    if count < 0:
        return rows[:]
    if count > len(rows):
        raise ValueError(f"requested {count} rows, only found {len(rows)}")
    rows = rows[:]
    rng.shuffle(rows)
    return rows[:count]


def normalize_single_rows(rows: list[dict], image_root: str, source_name: str) -> list[dict]:
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
                    "source": source_name,
                    "family": row.get("meta", {}).get("family"),
                    "formulas": [block["formula"] for block in payload["blocks"]],
                    "layout_lite": True,
                },
            }
        )
    return out


def normalize_lite_rows(rows: list[dict], source_name: str) -> list[dict]:
    out = []
    for row in rows:
        payload = lite_payload(parse_payload(row["messages"][1]["content"]))
        normalized = {
            "messages": [
                {"role": "user", "content": "<image>Chemical Structure Recognition:"},
                {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            "images": row["images"],
            "meta": {
                **row.get("meta", {}),
                "source": source_name,
                "formulas": [block["formula"] for block in payload["blocks"]],
                "layout_lite": True,
            },
        }
        out.append(normalized)
    return out


def parse_multi_source(spec: str) -> dict:
    parts = spec.split(",")
    if len(parts) != 5:
        raise ValueError("--multi-source must be NAME,SFT_DIR,TRAIN,EVAL,TEST")
    return {
        "name": parts[0],
        "sft_dir": Path(parts[1]),
        "train": int(parts[2]),
        "eval": int(parts[3]),
        "test": int(parts[4]),
    }


def parse_single_source(spec: str) -> dict:
    parts = spec.split(",")
    if len(parts) != 6:
        raise ValueError("--single-source must be NAME,SFT_DIR,IMAGE_ROOT,TRAIN,EVAL,TEST")
    return {
        "name": parts[0],
        "sft_dir": Path(parts[1]),
        "image_root": parts[2],
        "train": int(parts[3]),
        "eval": int(parts[4]),
        "test": int(parts[5]),
    }


def split_counts(rows: list[dict]) -> Counter:
    counts: Counter[int] = Counter()
    for row in rows:
        counts[len(parse_payload(row["messages"][1]["content"]).get("blocks", []))] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix several lite SFT datasets while preserving source tags.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--multi-source", action="append", default=[], help="NAME,SFT_DIR,TRAIN,EVAL,TEST")
    parser.add_argument("--single-source", action="append", default=[], help="NAME,SFT_DIR,IMAGE_ROOT,TRAIN,EVAL,TEST")
    parser.add_argument("--seed", type=int, default=2026051012)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    multi_sources = [parse_multi_source(spec) for spec in args.multi_source]
    single_sources = [parse_single_source(spec) for spec in args.single_source]
    splits = {"train": [], "eval": [], "test": []}
    source_summary = {}

    for source in multi_sources:
        source_summary[source["name"]] = {}
        for split in ["train", "eval", "test"]:
            rows = sample_rows(read_jsonl(source["sft_dir"] / f"{split}.jsonl"), source[split], rng)
            rows = normalize_lite_rows(rows, source["name"])
            splits[split].extend(rows)
            source_summary[source["name"]][split] = len(rows)

    for source in single_sources:
        source_summary[source["name"]] = {}
        for split in ["train", "eval", "test"]:
            rows = sample_rows(read_jsonl(source["sft_dir"] / f"{split}.jsonl"), source[split], rng)
            rows = normalize_single_rows(rows, source["image_root"], source["name"])
            splits[split].extend(rows)
            source_summary[source["name"]][split] = len(rows)

    for split, rows in splits.items():
        rng.shuffle(rows)
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", rows)
        if split == "train":
            write_jsonl(output_dir / "sft" / "smoke_train.jsonl", rows[:8])
        elif split == "eval":
            write_jsonl(output_dir / "sft" / "smoke_eval.jsonl", rows[:8])

    summary = {
        "seed": args.seed,
        "target": "mixed lite: bbox + formula + elemental_counts",
        "sources": source_summary,
        "splits": {
            split: {
                "pages": len(rows),
                "block_count_distribution": {str(k): v for k, v in sorted(split_counts(rows).items())},
                "file": str(output_dir / "sft" / f"{split}.jsonl"),
            }
            for split, rows in splits.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

