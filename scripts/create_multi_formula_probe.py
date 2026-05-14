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


def parse_payload(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("assistant message does not contain a JSON object")
    return json.loads(text[start : end + 1])


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


def resolve_image(dataset_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return dataset_dir / path


def crop_block(image: Image.Image, bbox: list[int], padding: int) -> Image.Image:
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(image.width, x1 + padding)
    y1 = min(image.height, y1 + padding)
    return image.crop((x0, y0, x1, y1))


def fit_image(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / image.width, max_h / image.height)
    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def source_block(row: dict) -> dict:
    payload = parse_payload(row["messages"][1]["content"])
    blocks = [block for block in payload.get("blocks", []) if block.get("type") == "chemical_structure"]
    if len(blocks) != 1:
        raise ValueError(f"expected one chemical_structure block for {row.get('meta', {}).get('id')}")
    return blocks[0]


def make_probe_page(
    rows: list[dict],
    dataset_dir: Path,
    output_image: Path,
    rng: random.Random,
) -> dict:
    canvas = Image.new("RGB", (1400, 980), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = find_font(26)
    small_font = find_font(18)
    micro_font = find_font(14)

    draw.text((44, 30), "Organic chemistry extraction sheet", fill=(35, 35, 35), font=title_font)
    draw.text((44, 68), "Ignore reference text: C6H12O6 / NaCl / H2O / XYZ-4096", fill=(122, 122, 122), font=small_font)
    draw.line((48, 110, 1352, 110), fill=(210, 210, 210), width=2)
    draw.line((700, 132, 700, 918), fill=(228, 228, 228), width=2)
    draw.line((48, 508, 1352, 508), fill=(228, 228, 228), width=2)

    slots = [
        (90, 150, 500, 290),
        (800, 145, 500, 295),
        (100, 585, 500, 290),
        (790, 575, 520, 305),
    ]
    distractors = [
        (522, 172, "not target: C12H22O11"),
        (1016, 452, "ref. pKa=4.76"),
        (78, 520, "ignore line: NaCl + H2O"),
        (1086, 832, "XYZ-4096"),
        (500, 930, "batch note: R1, R2, R3 are placeholders"),
    ]
    for x, y, text in distractors:
        draw.text((x, y), text, fill=(145, 145, 145), font=micro_font)

    blocks = []
    for idx, (row, slot) in enumerate(zip(rows, slots), start=1):
        block = source_block(row)
        image = Image.open(resolve_image(dataset_dir, row["images"][0])).convert("RGB")
        crop = fit_image(crop_block(image, block["bbox"], padding=18), slot[2], slot[3])
        x = slot[0] + (slot[2] - crop.width) // 2 + rng.randint(-12, 12)
        y = slot[1] + (slot[3] - crop.height) // 2 + rng.randint(-10, 10)
        canvas.paste(crop, (x, y))
        draw.text((slot[0], slot[1] - 26), f"S{idx}", fill=(80, 80, 80), font=small_font)
        blocks.append(
            {
                "type": "chemical_structure",
                "text": "",
                "bbox": [x, y, x + crop.width, y + crop.height],
                "formula": block["formula"],
                "elemental_counts": block.get("elemental_counts", {}),
                "smiles": block.get("smiles", ""),
                "name": block.get("name", ""),
                "name_cn": block.get("name_cn", ""),
            }
        )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_image, optimize=True)
    return {"blocks": blocks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create multi-molecule recombination probes from SFT formula rows.")
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--page-count", type=int, default=10)
    parser.add_argument("--per-page", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026050923)
    args = parser.parse_args()

    if args.per_page != 4:
        raise ValueError("This probe currently uses a fixed 2x2 layout, so --per-page must be 4.")

    rng = random.Random(args.seed)
    rows = read_jsonl(Path(args.source_jsonl))
    rng.shuffle(rows)
    need = args.page_count * args.per_page
    if len(rows) < need:
        raise RuntimeError(f"Need {need} rows, found {len(rows)}.")

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    out_rows = []
    summary = []
    for page_idx in range(args.page_count):
        page_rows = rows[page_idx * args.per_page : (page_idx + 1) * args.per_page]
        image_name = f"probe_{page_idx + 1:04d}.png"
        payload = make_probe_page(
            page_rows,
            Path(args.dataset_dir),
            images_dir / image_name,
            rng,
        )
        out_rows.append(
            {
                "messages": [
                    {"role": "user", "content": "<image>Chemical Structure Recognition:"},
                    {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                ],
                "images": [f"images/{image_name}"],
                "meta": {
                    "id": f"probe_{page_idx + 1:04d}",
                    "variant": "multi_formula_probe",
                    "formulas": [block["formula"] for block in payload["blocks"]],
                },
            }
        )
        summary.append(out_rows[-1]["meta"])

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "test.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps({"pages": args.page_count, "per_page": args.per_page, "seed": args.seed, "samples": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"jsonl": str(jsonl_path), "pages": args.page_count, "per_page": args.per_page}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

