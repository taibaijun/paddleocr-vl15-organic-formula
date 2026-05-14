import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


NOISE_TEXTS = [
    "not target: NaCl + H2O",
    "C6H12O6 ???",
    "OCR_TEST_2026",
    "XYZ-4096",
    "$ % # @ 7 8 9",
    "ignore this line",
    "batch id: A-17",
    "\u8bf7\u5ffd\u7565\u8fd9\u884c\u6587\u672c",
    "\u5e72\u6270\u5b57\u7b26",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf" if mono else None,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def text_bbox(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font) -> list[int]:
    box = draw.textbbox(xy, text, font=font)
    return [int(v) for v in box]


def resize_for_width(image: Image.Image, width: int) -> tuple[Image.Image, float, float]:
    ratio = width / image.width
    height = max(1, int(image.height * ratio))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    return resized, ratio, height / image.height


def transform_bbox(bbox: list[int], sx: float, sy: float, dx: int, dy: int) -> list[int]:
    return [
        int(bbox[0] * sx + dx),
        int(bbox[1] * sy + dy),
        int(bbox[2] * sx + dx),
        int(bbox[3] * sy + dy),
    ]


def paste_source(
    canvas: Image.Image,
    row: dict,
    dataset_root: Path,
    x: int,
    y: int,
    width: int,
    alpha: int,
    include_all_blocks: bool,
) -> list[dict]:
    src = Image.open(dataset_root / row["image_path"]).convert("RGBA")
    resized, sx, sy = resize_for_width(src, width)
    if alpha < 255:
        resized.putalpha(Image.new("L", resized.size, alpha))
    canvas.alpha_composite(resized, (x, y))

    blocks = []
    for block in row.get("layout_blocks", []):
        if not include_all_blocks and block.get("type") != "chemical_structure":
            continue
        out = dict(block)
        if "bbox" in out:
            out["bbox"] = transform_bbox(out["bbox"], sx, sy, x, y)
        if out.get("type") == "chemical_structure":
            out.setdefault("name", row["name"])
            out.setdefault("name_cn", row["name_cn"])
            out.setdefault("formula", row["formula"])
            out.setdefault("smiles", row["canonical_smiles"])
        blocks.append(out)
    return blocks


def add_noise(
    canvas: Image.Image,
    rng: random.Random,
    count: int,
    include_blocks: bool,
) -> list[dict]:
    draw = ImageDraw.Draw(canvas)
    blocks = []
    for _ in range(count):
        text = rng.choice(NOISE_TEXTS)
        font = load_font(rng.randint(18, 30), mono=rng.random() < 0.35)
        x = rng.randint(18, max(18, canvas.width - 320))
        y = rng.randint(16, max(16, canvas.height - 42))
        color = rng.choice(["#b91c1c", "#1d4ed8", "#047857", "#7c3aed", "#111827", "#dc2626"])
        draw.text((x, y), text, fill=color, font=font)
        if include_blocks:
            blocks.append({"type": "noise_text", "text": text, "bbox": text_bbox(draw, (x, y), text, font)})
    return blocks


def draw_distractors(canvas: Image.Image, rng: random.Random) -> None:
    draw = ImageDraw.Draw(canvas)
    for _ in range(rng.randint(1, 4)):
        color = rng.choice(["#9ca3af", "#d1d5db", "#cbd5e1"])
        if rng.random() < 0.5:
            x1 = rng.randint(10, canvas.width - 80)
            y1 = rng.randint(10, canvas.height - 80)
            x2 = rng.randint(x1 + 40, min(canvas.width - 10, x1 + 520))
            y2 = rng.randint(y1 + 30, min(canvas.height - 10, y1 + 360))
            draw.rectangle((x1, y1, x2, y2), outline=color, width=rng.randint(1, 3))
        else:
            draw.line(
                (
                    rng.randint(0, canvas.width),
                    rng.randint(0, canvas.height),
                    rng.randint(0, canvas.width),
                    rng.randint(0, canvas.height),
                ),
                fill=color,
                width=rng.randint(1, 3),
            )


def make_hard_row(
    rows: list[dict],
    caption_rows: list[dict],
    dataset_root: Path,
    output_dir: Path,
    index: int,
    rng: random.Random,
) -> dict:
    mode = rng.choices(
        ["overlay_structure", "multi_panel_structure", "noisy_single_structure", "noisy_caption"],
        weights=[0.34, 0.28, 0.22, 0.16],
        k=1,
    )[0]
    canvas_size = rng.choice([(1100, 760), (1280, 900), (1180, 820), (1024, 768)])
    canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 255))
    blocks: list[dict] = []

    if mode == "noisy_caption":
        source = rng.choice(caption_rows or rows)
        x = rng.randint(24, 70)
        y = rng.randint(40, 100)
        width = rng.randint(780, min(1000, canvas.width - x - 20))
        blocks.extend(paste_source(canvas, source, dataset_root, x, y, width, 255, True))
        draw_distractors(canvas, rng)
        blocks.extend(add_noise(canvas, rng, rng.randint(3, 7), include_blocks=True))
        variant = "caption"
        sources = [source]
    elif mode == "noisy_single_structure":
        source = rng.choice(rows)
        x = rng.randint(50, 160)
        y = rng.randint(50, 150)
        width = rng.randint(700, min(980, canvas.width - x - 20))
        blocks.extend(paste_source(canvas, source, dataset_root, x, y, width, 255, False))
        draw_distractors(canvas, rng)
        add_noise(canvas, rng, rng.randint(4, 9), include_blocks=False)
        variant = "structure"
        sources = [source]
    else:
        sample_count = 3 if mode == "multi_panel_structure" and rng.random() < 0.35 else 2
        sources = rng.sample(rows, sample_count)
        for slot, source in enumerate(sources):
            if mode == "overlay_structure":
                x = rng.randint(40 + slot * 160, 270 + slot * 180)
                y = rng.randint(40, 180 + slot * 80)
                width = rng.randint(680, 880)
                alpha = 170 if slot > 0 else 245
            else:
                col = slot % 2
                row_idx = slot // 2
                x = 30 + col * (canvas.width // 2) + rng.randint(0, 30)
                y = 30 + row_idx * (canvas.height // 2) + rng.randint(0, 30)
                max_width = max(360, min(600, canvas.width // 2 - 50))
                min_width = min(470, max_width)
                width = rng.randint(min_width, max_width)
                alpha = 255
            blocks.extend(paste_source(canvas, source, dataset_root, x, y, width, alpha, False))
        draw_distractors(canvas, rng)
        add_noise(canvas, rng, rng.randint(4, 8), include_blocks=False)
        variant = "structure"

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    item_id = f"{index:06d}_{mode}"
    image_path = image_dir / f"{item_id}.png"
    canvas.convert("RGB").save(image_path, optimize=True)

    text_blocks = [block["text"] for block in blocks if block.get("text")]
    formulas = [source.get("formula", "") for source in sources]
    smiles = [source.get("canonical_smiles", "") for source in sources]
    names = [source.get("name", "") for source in sources]
    names_cn = [source.get("name_cn", "") for source in sources]

    return {
        "id": item_id,
        "variant": variant,
        "layout": mode,
        "image_path": str(image_path),
        "svg_path": "",
        "name": " + ".join(names),
        "name_cn": " + ".join(names_cn),
        "formula": " | ".join(formulas),
        "canonical_smiles": " | ".join(smiles),
        "molecular_weight": round(sum(float(source.get("molecular_weight", 0.0)) for source in sources), 4),
        "ocr_text": "\n".join(text_blocks) if variant == "caption" else "",
        "layout_blocks": blocks,
        "targets": {
            "name": " + ".join(names),
            "name_cn": " + ".join(names_cn),
            "formula": " | ".join(formulas),
            "canonical_smiles": " | ".join(smiles),
            "layout_blocks": blocks,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Append hard composite samples to an organic OCR dataset.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir)
    dataset_root = Path(".")
    rng = random.Random(args.seed)

    base_rows = read_jsonl(labels_path)
    caption_rows = [row for row in base_rows if row.get("variant") == "caption"]
    start_index = len(base_rows) + 1
    hard_rows = []

    for offset in range(args.count):
        hard_rows.append(
            make_hard_row(base_rows, caption_rows, dataset_root, output_dir, start_index + offset, rng)
        )
        if args.progress_every and (offset + 1) % args.progress_every == 0:
            print(f"hard samples {offset + 1}/{args.count}", flush=True)

    all_rows = base_rows + hard_rows
    write_jsonl(labels_path, all_rows)
    summary = {
        "base_count": len(base_rows),
        "hard_count": len(hard_rows),
        "total": len(all_rows),
        "labels_jsonl": str(labels_path),
        "hard_modes": {
            mode: sum(1 for row in hard_rows if row["layout"] == mode)
            for mode in sorted({row["layout"] for row in hard_rows})
        },
    }
    (output_dir / "hard_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

