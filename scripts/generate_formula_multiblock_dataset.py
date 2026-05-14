import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


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


def source_block(row: dict) -> dict:
    payload = parse_payload(row["messages"][1]["content"])
    blocks = [block for block in payload.get("blocks", []) if block.get("type") == "chemical_structure"]
    if len(blocks) != 1:
        raise ValueError(f"expected one chemical_structure block for {row.get('meta', {}).get('id')}")
    return blocks[0]


def resolve_image(dataset_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return dataset_dir / path


def find_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/mnt/c/Windows/Fonts/arial.ttf",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def weighted_block_count(rng: random.Random) -> int:
    # Keep some single-object rehearsal, but mostly train multi-block page behavior.
    choices = [(1, 18), (2, 32), (3, 25), (4, 25)]
    total = sum(weight for _, weight in choices)
    pick = rng.randint(1, total)
    upto = 0
    for value, weight in choices:
        upto += weight
        if pick <= upto:
            return value
    return 4


def crop_structure(image: Image.Image, bbox: list[int], padding: int) -> Image.Image:
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(image.width, x1 + padding)
    y1 = min(image.height, y1 + padding)
    return image.crop((x0, y0, x1, y1))


def fit_image(image: Image.Image, max_w: int, max_h: int, rng: random.Random) -> Image.Image:
    scale = min(max_w / image.width, max_h / image.height)
    scale *= rng.uniform(0.86, 1.0)
    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    if rng.random() < 0.12:
        resized = resized.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.08, 0.25)))
    return resized


def slots_for(canvas_size: tuple[int, int], block_count: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    width, height = canvas_size
    top = rng.randint(96, 132)
    bottom_margin = rng.randint(44, 70)
    usable_h = height - top - bottom_margin
    left = rng.randint(42, 76)
    right = rng.randint(42, 76)
    gap = rng.randint(28, 52)

    if block_count == 1:
        return [(left + 70, top + 40, width - left - right - 140, usable_h - 80)]
    if block_count == 2:
        if rng.random() < 0.55:
            slot_w = (width - left - right - gap) // 2
            return [(left, top, slot_w, usable_h), (left + slot_w + gap, top, slot_w, usable_h)]
        slot_h = (usable_h - gap) // 2
        return [(left + 90, top, width - left - right - 180, slot_h), (left + 90, top + slot_h + gap, width - left - right - 180, slot_h)]
    if block_count == 3:
        slot_w = (width - left - right - gap) // 2
        slot_h = (usable_h - gap) // 2
        if rng.random() < 0.5:
            return [
                (left, top, slot_w, slot_h),
                (left + slot_w + gap, top, slot_w, slot_h),
                (left + width // 4, top + slot_h + gap, width // 2, slot_h),
            ]
        return [
            (left, top + usable_h // 5, slot_w, slot_h),
            (left + slot_w + gap, top, slot_w, slot_h),
            (left + slot_w + gap, top + slot_h + gap, slot_w, slot_h),
        ]

    slot_w = (width - left - right - gap) // 2
    slot_h = (usable_h - gap) // 2
    return [
        (left, top, slot_w, slot_h),
        (left + slot_w + gap, top, slot_w, slot_h),
        (left, top + slot_h + gap, slot_w, slot_h),
        (left + slot_w + gap, top + slot_h + gap, slot_w, slot_h),
    ]


def draw_page_chrome(canvas: Image.Image, rng: random.Random, block_count: int) -> None:
    draw = ImageDraw.Draw(canvas)
    title_font = find_font(rng.randint(22, 28))
    note_font = find_font(rng.randint(13, 18))
    micro_font = find_font(rng.randint(11, 15))
    width, height = canvas.size

    titles = [
        "Organic chemistry extraction sheet",
        "Reaction intermediate screening",
        "Compound structure panel",
        "Supplementary molecule list",
    ]
    draw.text((rng.randint(30, 54), rng.randint(22, 38)), rng.choice(titles), fill=(35, 35, 35), font=title_font)
    if rng.random() < 0.82:
        draw.text(
            (rng.randint(34, 58), rng.randint(62, 78)),
            rng.choice(
                [
                    "Ignore reference text: C6H12O6 / NaCl / H2O / XYZ-4096",
                    "Reference only: pKa=4.76, sample=A-17, not a formula target",
                    "Batch note: R1, R2 and R3 are placeholders",
                ]
            ),
            fill=(118, 118, 118),
            font=note_font,
        )
    if rng.random() < 0.75:
        y = rng.randint(100, 126)
        draw.line((38, y, width - 38, y), fill=(rng.randint(196, 224),) * 3, width=rng.randint(1, 2))
    if block_count >= 3 and rng.random() < 0.72:
        draw.line((width // 2, 132, width // 2, height - 52), fill=(224, 224, 224), width=1)
        draw.line((48, height // 2, width - 48, height // 2), fill=(224, 224, 224), width=1)

    distractors = [
        "not target: C12H22O11",
        "ignore: NaCl + H2O",
        "OCR_TEST_2026",
        "$ % # @ 7 8 9",
        "XYZ-4096",
        "R = Me, Et, Ph",
        "calc. mass pending",
        "solvent: DMSO",
    ]
    for _ in range(rng.randint(2, 7)):
        text = rng.choice(distractors)
        x = rng.randint(24, max(24, width - 260))
        y = rng.randint(92, max(92, height - 34))
        draw.text((x, y), text, fill=(rng.randint(125, 170),) * 3, font=micro_font)

    if rng.random() < 0.2:
        x0 = rng.randint(10, width - 180)
        y0 = rng.randint(110, height - 140)
        x1 = min(width - 12, x0 + rng.randint(120, 340))
        y1 = min(height - 12, y0 + rng.randint(60, 220))
        draw.rectangle((x0, y0, x1, y1), outline=(210, 210, 210), width=1)


def make_page(
    page_id: str,
    sampled_rows: list[dict],
    source_dataset_dir: Path,
    output_image: Path,
    rng: random.Random,
) -> tuple[dict, dict]:
    canvas_size = rng.choice([(1050, 760), (1180, 820), (1280, 900), (1400, 980)])
    bg = rng.randint(248, 255)
    canvas = Image.new("RGB", canvas_size, (bg, bg, bg))
    draw_page_chrome(canvas, rng, len(sampled_rows))
    draw = ImageDraw.Draw(canvas)
    label_font = find_font(rng.randint(14, 18))
    slots = slots_for(canvas_size, len(sampled_rows), rng)

    blocks = []
    source_ids = []
    for idx, (row, slot) in enumerate(zip(sampled_rows, slots), start=1):
        block = source_block(row)
        source_ids.append(row.get("meta", {}).get("id", ""))
        image = Image.open(resolve_image(source_dataset_dir, row["images"][0])).convert("RGB")
        crop = crop_structure(image, block["bbox"], padding=rng.randint(12, 24))
        fitted = fit_image(crop, slot[2], slot[3], rng)
        x = slot[0] + (slot[2] - fitted.width) // 2 + rng.randint(-16, 16)
        y = slot[1] + (slot[3] - fitted.height) // 2 + rng.randint(-14, 14)
        x = max(0, min(canvas.width - fitted.width, x))
        y = max(0, min(canvas.height - fitted.height, y))
        canvas.paste(fitted, (x, y))
        if rng.random() < 0.74:
            draw.text((max(6, slot[0] + 4), max(6, slot[1] - 24)), f"S{idx}", fill=(78, 78, 78), font=label_font)
        blocks.append(
            {
                "type": "chemical_structure",
                "text": "",
                "bbox": [x, y, x + fitted.width, y + fitted.height],
                "formula": block["formula"],
                "elemental_counts": block.get("elemental_counts", {}),
                "smiles": block.get("smiles", ""),
                "name": block.get("name", ""),
                "name_cn": block.get("name_cn", ""),
            }
        )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_image, optimize=True)
    payload = {"blocks": blocks}
    raw = {
        "id": page_id,
        "variant": f"{len(sampled_rows)}_structure_page",
        "layout": "formula_multiblock_curriculum",
        "image_path": str(output_image),
        "source_ids": source_ids,
        "formulas": [block["formula"] for block in blocks],
        "layout_blocks": blocks,
        "targets": payload,
    }
    sft = {
        "messages": [
            {"role": "user", "content": "<image>Chemical Structure Recognition:"},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        "images": [f"images/{output_image.name}"],
        "meta": {
            "id": page_id,
            "variant": raw["variant"],
            "formulas": raw["formulas"],
            "source_ids": source_ids,
        },
    }
    return raw, sft


def build_split(
    split: str,
    source_rows: list[dict],
    source_dataset_dir: Path,
    output_dir: Path,
    page_count: int,
    seed: int,
) -> tuple[list[dict], list[dict], Counter]:
    rng = random.Random(seed)
    raw_rows = []
    sft_rows = []
    distribution: Counter[int] = Counter()
    for index in range(1, page_count + 1):
        block_count = weighted_block_count(rng)
        sampled_rows = rng.sample(source_rows, block_count)
        page_id = f"{split}_{index:06d}"
        output_image = output_dir / "sft" / "images" / f"{page_id}.png"
        raw, sft = make_page(page_id, sampled_rows, source_dataset_dir, output_image, rng)
        raw_rows.append(raw)
        sft_rows.append(sft)
        distribution[block_count] += 1
        if index % 1000 == 0:
            print(f"rendered {split}: {index}/{page_count}", flush=True)
    return raw_rows, sft_rows, distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 1-4 block formula page SFT data from a single-molecule pool.")
    parser.add_argument("--source-root", required=True, help="Directory containing train/eval/test JSONL and images.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=24000)
    parser.add_argument("--eval-count", type=int, default=1600)
    parser.add_argument("--test-count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=2026050924)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    split_specs = [
        ("train", args.train_count, args.seed + 11),
        ("eval", args.eval_count, args.seed + 29),
        ("test", args.test_count, args.seed + 47),
    ]
    summary = {
        "seed": args.seed,
        "source_root": str(source_root),
        "target": "1-4 chemical_structure blocks with formula, elemental_counts, canonical SMILES and bbox",
        "splits": {},
    }

    canonical_sets: dict[str, set[str]] = {}
    for split, count, seed in split_specs:
        source_rows = read_jsonl(source_root / f"{split}.jsonl")
        canonical_sets[split] = {row.get("meta", {}).get("canonical_smiles", "") for row in source_rows}
        raw_rows, sft_rows, distribution = build_split(split, source_rows, source_root, output_dir, count, seed)
        write_jsonl(output_dir / "raw" / f"{split}_labels.jsonl", raw_rows)
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", sft_rows)
        if split == "train":
            write_jsonl(output_dir / "sft" / "smoke_train.jsonl", sft_rows[:8])
        elif split == "eval":
            write_jsonl(output_dir / "sft" / "smoke_eval.jsonl", sft_rows[:8])
        summary["splits"][split] = {
            "pages": len(sft_rows),
            "block_count_distribution": {str(k): distribution[k] for k in sorted(distribution)},
            "total_blocks": sum(k * distribution[k] for k in distribution),
            "file": str(output_dir / "sft" / f"{split}.jsonl"),
        }

    summary["source_overlap"] = {
        "train_eval": len(canonical_sets["train"] & canonical_sets["eval"]),
        "train_test": len(canonical_sets["train"] & canonical_sets["test"]),
        "eval_test": len(canonical_sets["eval"] & canonical_sets["test"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

