import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/mnt/d/百度ocr比赛")
TRAIN_JSON = ROOT / "output/organic_paddleformers_sft_5000/train.jsonl"
DATA_DIR = ROOT / "output/organic_paddleformers_sft_5000"
OUT_DIR = ROOT / "output/stress_tests"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def read_first_rows(count: int) -> list[dict]:
    rows = []
    with TRAIN_JSON.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= count:
                break
    return rows


def main() -> None:
    rows = read_first_rows(2)
    first, second = rows
    first_path = DATA_DIR / first["images"][0]
    second_path = DATA_DIR / second["images"][0]

    first_image = Image.open(first_path).convert("RGBA")
    second_image = Image.open(second_path).convert("RGBA")

    canvas = Image.new("RGBA", (1100, 760), (255, 255, 255, 255))
    first_image = first_image.resize(
        (760, int(first_image.height * 760 / first_image.width)),
        Image.Resampling.LANCZOS,
    )
    second_image = second_image.resize(
        (800, int(second_image.height * 800 / second_image.width)),
        Image.Resampling.LANCZOS,
    )

    canvas.alpha_composite(first_image, (80, 160))
    second_image.putalpha(Image.new("L", second_image.size, 150))
    canvas.alpha_composite(second_image, (250, 60))

    draw = ImageDraw.Draw(canvas)
    font = load_font(26)
    noise_items = [
        ("干扰字符 XYZ-4096", (38, 36), "#b91c1c"),
        ("not target: NaCl + H2O", (730, 35), "#1d4ed8"),
        ("OCR_TEST_2026", (30, 690), "#047857"),
        ("$ % # @ 7 8 9", (780, 690), "#7c3aed"),
        ("随机文本：请忽略", (455, 710), "#111827"),
        ("C6H12O6 ???", (50, 105), "#dc2626"),
    ]
    for text, xy, color in noise_items:
        draw.text(xy, text, fill=color, font=font)

    draw.rectangle((40, 120, 1010, 700), outline="#9ca3af", width=2)
    draw.line((30, 420, 1050, 180), fill="#6b7280", width=2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = OUT_DIR / "overlay_two_train_images_with_noise.png"
    canvas.convert("RGB").save(image_path, optimize=True)

    meta = {
        "image": str(image_path),
        "source_a": {"path": str(first_path), "meta": first["meta"]},
        "source_b": {"path": str(second_path), "meta": second["meta"]},
        "noise_text": [item[0] for item in noise_items],
    }
    meta_path = OUT_DIR / "overlay_two_train_images_with_noise_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

