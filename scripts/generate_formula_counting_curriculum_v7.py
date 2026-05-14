import argparse
import json
import random
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from generate_formula_diverse_dataset import generate_records


ELEMENTS = ["C", "H", "N", "O", "S", "F", "Cl", "Br"]
VARIANTS = ["skeletal", "carbon_labeled", "explicit_methyl", "h_notes", "explicit_h"]


def load_font(size: int) -> ImageFont.ImageFont:
    for raw in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/mnt/c/Windows/Fonts/arial.ttf",
    ]:
        path = Path(raw)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def record_counts(record) -> dict[str, int]:
    return {element: int(record.elemental_counts[element]) for element in ELEMENTS if element in record.elemental_counts}


def draw_molecule(record, variant: str, rng: random.Random) -> Image.Image:
    mol = Chem.MolFromSmiles(record.canonical_smiles)
    if mol is None:
        raise ValueError(record.canonical_smiles)

    if variant == "explicit_h":
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.Mol(mol)

    if variant == "h_notes":
        for atom in mol.GetAtoms():
            h_count = atom.GetTotalNumHs()
            if atom.GetSymbol() == "C":
                atom.SetProp("atomNote", f"H{h_count}" if h_count else "H0")
            elif h_count:
                atom.SetProp("atomNote", f"H{h_count}")

    AllChem.Compute2DCoords(mol)
    size = rng.choice([(520, 380), (580, 420), (640, 460), (700, 500)])
    drawer = Draw.MolDraw2DCairo(size[0], size[1])
    options = drawer.drawOptions()
    options.padding = rng.uniform(0.05, 0.14)
    options.bondLineWidth = rng.uniform(1.6, 3.2)
    options.fixedBondLength = rng.uniform(24, 42)
    options.rotate = rng.choice([0, 0, 0, 90, 180, 270])
    options.explicitMethyl = variant in {"explicit_methyl", "carbon_labeled", "h_notes"}
    if variant in {"carbon_labeled", "h_notes"}:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "C":
                options.atomLabels[atom.GetIdx()] = "C"

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    image = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")
    if rng.random() < 0.16:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.08, 0.28)))
    return image


def compose_single(record, variant: str, rng: random.Random) -> tuple[Image.Image, list[int]]:
    mol_img = draw_molecule(record, variant, rng)
    canvas_size = rng.choice([(760, 560), (860, 640), (960, 700), (1040, 760)])
    bg = rng.randint(248, 255)
    canvas = Image.new("RGB", canvas_size, (bg, bg, bg))
    scale = min((canvas.width - 90) / mol_img.width, (canvas.height - 90) / mol_img.height)
    scale *= rng.uniform(0.82, 1.0)
    resized = mol_img.resize((max(1, int(mol_img.width * scale)), max(1, int(mol_img.height * scale))), Image.Resampling.LANCZOS)
    x = rng.randint(28, max(28, canvas.width - resized.width - 28))
    y = rng.randint(28, max(28, canvas.height - resized.height - 28))
    canvas.paste(resized, (x, y))
    draw = ImageDraw.Draw(canvas)
    if rng.random() < 0.25:
        font = load_font(rng.randint(13, 18))
        text = rng.choice(["ignore: NaCl + H2O", "ref only: C6H12O6", "batch A-17", "R = Me / Et"])
        draw.text((rng.randint(10, max(10, canvas.width - 190)), rng.randint(8, max(8, canvas.height - 28))), text, fill=(145, 145, 145), font=font)
    return canvas, [x, y, x + resized.width, y + resized.height]


def formula_target(blocks: list[dict]) -> str:
    slim = [{"formula": b["formula"], "elemental_counts": b["elemental_counts"]} for b in blocks]
    return json.dumps({"blocks": slim}, ensure_ascii=False, separators=(",", ":"))


def layout_target(blocks: list[dict]) -> str:
    return json.dumps({"blocks": blocks}, ensure_ascii=False, separators=(",", ":"))


def sft_row(image_path: Path, blocks: list[dict], row_id: str, variant: str, formula_only: bool) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "<image>Chemical Structure Recognition:"},
            {"role": "assistant", "content": formula_target(blocks) if formula_only else layout_target(blocks)},
        ],
        "images": [str(image_path)],
        "meta": {
            "id": row_id,
            "variant": variant,
            "formulas": [block["formula"] for block in blocks],
            "counting_curriculum_v7": True,
            "formula_only": formula_only,
        },
    }


def make_single_rows(records, split: str, image_dir: Path, rng: random.Random, start: int, train: bool) -> tuple[list[dict], list[dict], list[dict]]:
    formula_rows, layout_rows, raw_rows = [], [], []
    variant_choices = VARIANTS if train else ["skeletal"]
    repeats = 3 if train else 1
    idx = start
    for record in records:
        for _ in range(repeats):
            variant = rng.choice(variant_choices)
            image, bbox = compose_single(record, variant, rng)
            row_id = f"{split}_{idx:06d}_{variant}"
            image_path = image_dir / f"{row_id}.png"
            image.save(image_path, optimize=True)
            block = {
                "type": "chemical_structure",
                "text": "",
                "bbox": bbox,
                "formula": record.formula,
                "elemental_counts": record_counts(record),
                "smiles": record.canonical_smiles,
                "name": f"CountCurr {idx:06d}",
                "name_cn": f"CountCurr {idx:06d}",
            }
            raw_rows.append({"id": row_id, "image_path": str(image_path), "blocks": [block], "variant": variant})
            formula_rows.append(sft_row(image_path, [block], row_id, variant, True))
            layout_rows.append(sft_row(image_path, [block], row_id, variant, False))
            idx += 1
    return formula_rows, layout_rows, raw_rows


def slots_for(size: tuple[int, int], n: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    w, h = size
    margin = rng.randint(36, 58)
    gap = rng.randint(26, 48)
    top = rng.randint(86, 122)
    usable_h = h - top - margin
    if n == 2 and rng.random() < 0.5:
        sw = (w - 2 * margin - gap) // 2
        return [(margin, top, sw, usable_h), (margin + sw + gap, top, sw, usable_h)]
    if n == 2:
        sh = (usable_h - gap) // 2
        return [(margin + 90, top, w - 2 * margin - 180, sh), (margin + 90, top + sh + gap, w - 2 * margin - 180, sh)]
    sw = (w - 2 * margin - gap) // 2
    sh = (usable_h - gap) // 2
    slots = [(margin, top, sw, sh), (margin + sw + gap, top, sw, sh), (margin, top + sh + gap, sw, sh), (margin + sw + gap, top + sh + gap, sw, sh)]
    return slots[:n]


def make_multi_rows(raw_singles: list[dict], split: str, image_dir: Path, rng: random.Random, count: int, start: int) -> tuple[list[dict], list[dict]]:
    formula_rows, layout_rows = [], []
    for i in range(count):
        n = rng.choices([2, 3, 4], weights=[28, 32, 40], k=1)[0]
        samples = rng.sample(raw_singles, n)
        size = rng.choice([(1100, 780), (1280, 900), (1400, 980)])
        bg = rng.randint(248, 255)
        canvas = Image.new("RGB", size, (bg, bg, bg))
        draw = ImageDraw.Draw(canvas)
        title_font = load_font(rng.randint(20, 27))
        small_font = load_font(rng.randint(12, 16))
        draw.text((rng.randint(34, 56), rng.randint(22, 36)), rng.choice(["Compound structure panel", "Organic extraction sheet", "Molecule count worksheet"]), fill=(35, 35, 35), font=title_font)
        draw.text((rng.randint(36, 62), rng.randint(58, 78)), rng.choice(["ignore: C6H12O6 / NaCl / H2O", "reference text, not targets", "R groups and solvent notes are distractors"]), fill=(120, 120, 120), font=small_font)
        slots = slots_for(size, n, rng)
        blocks = []
        for sample, slot in zip(samples, slots):
            src = Image.open(sample["image_path"]).convert("RGB")
            block = sample["blocks"][0]
            x0, y0, x1, y1 = block["bbox"]
            crop = src.crop((max(0, x0 - 10), max(0, y0 - 10), min(src.width, x1 + 10), min(src.height, y1 + 10)))
            scale = min(slot[2] / crop.width, slot[3] / crop.height) * rng.uniform(0.82, 1.0)
            resized = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)
            px = slot[0] + (slot[2] - resized.width) // 2 + rng.randint(-12, 12)
            py = slot[1] + (slot[3] - resized.height) // 2 + rng.randint(-10, 10)
            canvas.paste(resized, (px, py))
            new_block = dict(block)
            new_block["bbox"] = [px, py, px + resized.width, py + resized.height]
            blocks.append(new_block)
        for _ in range(rng.randint(2, 6)):
            draw.text((rng.randint(20, size[0] - 230), rng.randint(92, size[1] - 34)), rng.choice(["not target: C12H22O11", "OCR_TEST_2026", "$ % # @ 7 8 9", "R = Me / Et / Ph"]), fill=(rng.randint(130, 170),) * 3, font=small_font)
        row_id = f"{split}_{start + i:06d}_multi{n}"
        image_path = image_dir / f"{row_id}.png"
        canvas.save(image_path, optimize=True)
        formula_rows.append(sft_row(image_path, blocks, row_id, f"{n}_structure_page", True))
        layout_rows.append(sft_row(image_path, blocks, row_id, f"{n}_structure_page", False))
    return formula_rows, layout_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-records", type=int, default=12000)
    parser.add_argument("--eval-records", type=int, default=350)
    parser.add_argument("--test-records", type=int, default=200)
    parser.add_argument("--train-multi", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=2026051307)
    args = parser.parse_args()

    out = Path(args.output_root)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    total = args.train_records + args.eval_records + args.test_records
    records = generate_records(total, args.seed)
    train_records = records[: args.train_records]
    eval_records = records[args.train_records : args.train_records + args.eval_records]
    test_records = records[args.train_records + args.eval_records :]
    splits = {
        "train": (train_records, True, args.train_multi),
        "eval": (eval_records, False, 0),
        "test": (test_records, False, 0),
    }
    summary = {"seed": args.seed, "splits": {}, "variants": VARIANTS}
    for split, (split_records, is_train, multi_count) in splits.items():
        rng = random.Random(args.seed + {"train": 11, "eval": 22, "test": 33}[split])
        f_rows, l_rows, raw = make_single_rows(split_records, split, image_dir, rng, 0, is_train)
        if multi_count:
            mf, ml = make_multi_rows(raw, split, image_dir, rng, multi_count, len(raw))
            f_rows.extend(mf)
            l_rows.extend(ml)
        write_jsonl(out / "formula" / "sft" / f"{split}.jsonl", f_rows)
        write_jsonl(out / "layout" / "sft" / f"{split}.jsonl", l_rows)
        if split == "train":
            write_jsonl(out / "formula" / "sft" / "smoke_train.jsonl", f_rows[:8])
            write_jsonl(out / "layout" / "sft" / "smoke_train.jsonl", l_rows[:8])
        if split == "eval":
            write_jsonl(out / "formula" / "sft" / "smoke_eval.jsonl", f_rows[:8])
            write_jsonl(out / "layout" / "sft" / "smoke_eval.jsonl", l_rows[:8])
        summary["splits"][split] = {"formula_rows": len(f_rows), "layout_rows": len(l_rows), "unique_records": len(split_records), "multi_pages": multi_count}
        print(json.dumps({"split": split, **summary["splits"][split]}, ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

