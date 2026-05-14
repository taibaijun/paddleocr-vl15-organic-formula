import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def numeric_id(sample_id: str) -> int:
    return int(sample_id.split("_", 1)[0])


def find_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def load_train_ids(train_jsonl: Path) -> set[str]:
    ids = set()
    for row in read_jsonl(train_jsonl):
        sample_id = row.get("meta", {}).get("id")
        if sample_id:
            ids.add(sample_id)
    return ids


def pick_samples(labels: list[dict], train_ids: set[str], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    candidates = []
    seen_formulas = set()
    for row in labels:
        blocks = [
            block
            for block in row.get("layout_blocks", [])
            if block.get("type") == "chemical_structure"
        ]
        if (
            row.get("id") in train_ids
            and row.get("variant") == "structure"
            and numeric_id(row["id"]) < 16000
            and len(blocks) == 1
            and row.get("formula") not in seen_formulas
        ):
            candidates.append(row)
            seen_formulas.add(row.get("formula"))

    rng.shuffle(candidates)
    if len(candidates) < count:
        raise RuntimeError(f"Only found {len(candidates)} candidates, need {count}.")
    return candidates[:count]


def crop_structure(row: dict, padding: int) -> Image.Image:
    image = Image.open(row["image_path"]).convert("RGB")
    block = next(block for block in row["layout_blocks"] if block.get("type") == "chemical_structure")
    x0, y0, x1, y1 = block["bbox"]
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(image.width, x1 + padding)
    y1 = min(image.height, y1 + padding)
    return image.crop((x0, y0, x1, y1))


def fit_image(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / image.width, max_h / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a recombined molecule-layout probe from training samples.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ids = load_train_ids(Path(args.train_jsonl))
    labels = read_jsonl(Path(args.labels_jsonl))
    samples = pick_samples(labels, train_ids, args.count, args.seed)

    canvas = Image.new("RGB", (1400, 980), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = find_font(26)
    small_font = find_font(18)
    micro_font = find_font(14)

    draw.text((46, 34), "Recombined chemistry probe", fill=(35, 35, 35), font=title_font)
    draw.text((46, 72), "Distractors: C6H12O6 / NaCl / H2O / XYZ-4096", fill=(120, 120, 120), font=small_font)
    draw.line((48, 108, 1352, 108), fill=(210, 210, 210), width=2)
    draw.line((700, 124, 700, 910), fill=(225, 225, 225), width=2)
    draw.line((48, 505, 1352, 505), fill=(225, 225, 225), width=2)

    slots = [
        (82, 148, 520, 285),
        (792, 140, 500, 305),
        (116, 580, 500, 300),
        (790, 560, 520, 315),
    ]
    noise = [
        (510, 172, "not target: C12H22O11"),
        (1020, 452, "ref. pKa=4.76"),
        (82, 516, "ignore line: NaCl + H2O"),
        (1086, 832, "XYZ-4096"),
    ]
    for x, y, text in noise:
        draw.text((x, y), text, fill=(150, 150, 150), font=micro_font)

    blocks = []
    for idx, (row, slot) in enumerate(zip(samples, slots), start=1):
        x, y, max_w, max_h = slot
        crop = fit_image(crop_structure(row, padding=16), max_w, max_h)
        offset_x = x + (max_w - crop.width) // 2
        offset_y = y + (max_h - crop.height) // 2
        canvas.paste(crop, (offset_x, offset_y))
        draw.text((x, y - 26), f"S{idx}", fill=(80, 80, 80), font=small_font)
        blocks.append(
            {
                "type": "chemical_structure",
                "text": "",
                "bbox": [offset_x, offset_y, offset_x + crop.width, offset_y + crop.height],
                "formula": row["formula"],
                "smiles": row["canonical_smiles"],
                "name": row["name"],
                "name_cn": row["name_cn"],
                "source_id": row["id"],
            }
        )

    image_path = output_dir / f"generalization_recombined_seed{args.seed}.png"
    target_path = output_dir / f"generalization_recombined_seed{args.seed}_target.json"
    canvas.save(image_path)
    target = {"blocks": blocks}
    target_path.write_text(json.dumps(target, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "image": str(image_path),
                "target": str(target_path),
                "formulas": [block["formula"] for block in blocks],
                "source_ids": [block["source_id"] for block in blocks],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

