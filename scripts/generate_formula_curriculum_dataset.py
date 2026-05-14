import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors


MAX_VALENCE = {
    "C": 4,
    "N": 3,
    "O": 2,
    "S": 2,
    "F": 1,
    "Cl": 1,
    "Br": 1,
}

ATOM_WEIGHTS = [
    ("C", 66),
    ("O", 14),
    ("N", 9),
    ("S", 3),
    ("F", 3),
    ("Cl", 3),
    ("Br", 2),
]


@dataclass(frozen=True)
class MoleculeRecord:
    canonical_smiles: str
    formula: str
    elemental_counts: dict[str, int]
    molecular_weight: float
    atom_count: int


def weighted_choice(rng: random.Random, weighted_items: list[tuple[str, int]]) -> str:
    total = sum(weight for _, weight in weighted_items)
    pick = rng.randint(1, total)
    upto = 0
    for item, weight in weighted_items:
        upto += weight
        if pick <= upto:
            return item
    return weighted_items[-1][0]


def bond_order_value(bond_type: Chem.BondType) -> int:
    if bond_type == Chem.BondType.DOUBLE:
        return 2
    if bond_type == Chem.BondType.TRIPLE:
        return 3
    return 1


def used_valence(mol: Chem.RWMol, atom_idx: int) -> int:
    return sum(bond_order_value(bond.GetBondType()) for bond in mol.GetAtomWithIdx(atom_idx).GetBonds())


def remaining_valence(mol: Chem.RWMol, atom_idx: int) -> int:
    atom = mol.GetAtomWithIdx(atom_idx)
    return MAX_VALENCE.get(atom.GetSymbol(), 0) - used_valence(mol, atom_idx)


def choose_bond(
    rng: random.Random,
    parent_symbol: str,
    child_symbol: str,
    parent_remaining: int,
) -> Chem.BondType | None:
    max_child = MAX_VALENCE[child_symbol]
    possible = [1]
    if parent_symbol == "C" and child_symbol in {"C", "O", "N"} and parent_remaining >= 2 and max_child >= 2:
        possible.append(2)
    if parent_symbol == "C" and child_symbol in {"C", "N"} and parent_remaining >= 3 and max_child >= 3:
        possible.append(3)

    weights = []
    for order in possible:
        if order == 1:
            weights.append((order, 84))
        elif order == 2:
            weights.append((order, 13))
        else:
            weights.append((order, 3))

    total = sum(weight for _, weight in weights)
    pick = rng.randint(1, total)
    upto = 0
    for order, weight in weights:
        upto += weight
        if pick <= upto:
            if order > parent_remaining or order > max_child:
                return None
            return {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
            }[order]
    return Chem.BondType.SINGLE


def random_tree_mol(rng: random.Random, atom_count: int) -> Chem.Mol | None:
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom("C"))
    for _ in range(atom_count - 1):
        parents = [idx for idx in range(mol.GetNumAtoms()) if remaining_valence(mol, idx) > 0]
        if not parents:
            return None
        parent_idx = rng.choice(parents)
        parent_symbol = mol.GetAtomWithIdx(parent_idx).GetSymbol()
        child_symbol = weighted_choice(rng, ATOM_WEIGHTS)
        if parent_symbol in {"F", "Cl", "Br"}:
            return None
        bond_type = choose_bond(rng, parent_symbol, child_symbol, remaining_valence(mol, parent_idx))
        if bond_type is None:
            return None
        child_idx = mol.AddAtom(Chem.Atom(child_symbol))
        mol.AddBond(parent_idx, child_idx, bond_type)

    try:
        out = mol.GetMol()
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def make_record(mol: Chem.Mol) -> MoleculeRecord | None:
    if Chem.GetFormalCharge(mol) != 0:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if not any(atom.GetSymbol() == "C" for atom in mol.GetAtoms()):
        return None

    canonical = Chem.MolToSmiles(mol, canonical=True)
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None
    atom_count = mol.GetNumHeavyAtoms()
    molecular_weight = Descriptors.MolWt(mol)
    if atom_count < 3 or atom_count > 18 or molecular_weight < 35 or molecular_weight > 280:
        return None
    formula = rdMolDescriptors.CalcMolFormula(mol)
    counts = Counter(atom.GetSymbol() for atom in Chem.AddHs(mol).GetAtoms())
    elemental_counts = {
        element: counts[element]
        for element in ["C", "H", "N", "O", "S", "F", "Cl", "Br"]
        if counts[element]
    }
    return MoleculeRecord(
        canonical_smiles=canonical,
        formula=formula,
        elemental_counts=elemental_counts,
        molecular_weight=round(molecular_weight, 4),
        atom_count=atom_count,
    )


def generate_records(total: int, seed: int) -> list[MoleculeRecord]:
    rng = random.Random(seed)
    records: dict[str, MoleculeRecord] = {}
    attempts = 0
    max_attempts = max(20000, total * 80)
    while len(records) < total and attempts < max_attempts:
        attempts += 1
        atom_count = rng.randint(3, 16)
        mol = random_tree_mol(rng, atom_count)
        if mol is None:
            continue
        record = make_record(mol)
        if record is None:
            continue
        records.setdefault(record.canonical_smiles, record)
        if len(records) % 1000 == 0 and len(records) > 0:
            print(f"unique molecules: {len(records)}/{total}", flush=True)
    if len(records) < total:
        raise RuntimeError(f"Generated only {len(records)} unique molecules after {attempts} attempts.")
    values = list(records.values())
    rng.shuffle(values)
    return values


def draw_mol_png(mol: Chem.Mol, size: tuple[int, int], rng: random.Random) -> Image.Image:
    AllChem.Compute2DCoords(mol)
    drawer = Draw.MolDraw2DCairo(size[0], size[1])
    options = drawer.drawOptions()
    options.padding = rng.uniform(0.05, 0.18)
    options.bondLineWidth = rng.uniform(1.5, 3.4)
    options.fixedBondLength = rng.uniform(25, 45)
    options.rotate = rng.choice([0, 0, 0, 90, 180, 270])
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    image = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")
    if rng.random() < 0.18:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.12, 0.4)))
    return image


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/mnt/c/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def compose_structure_image(
    record: MoleculeRecord,
    rng: random.Random,
) -> tuple[Image.Image, list[int]]:
    mol = Chem.MolFromSmiles(record.canonical_smiles)
    if mol is None:
        raise ValueError(record.canonical_smiles)

    canvas_size = rng.choice([(760, 560), (820, 620), (900, 640), (960, 720), (700, 520)])
    mol_size = (
        rng.randint(420, min(760, canvas_size[0] - 40)),
        rng.randint(300, min(520, canvas_size[1] - 40)),
    )
    mol_image = draw_mol_png(mol, mol_size, rng)
    bg_value = rng.randint(247, 255)
    canvas = Image.new("RGB", canvas_size, (bg_value, bg_value, bg_value))
    max_x = max(0, canvas.width - mol_image.width)
    max_y = max(0, canvas.height - mol_image.height)
    x = rng.randint(0, max_x) if max_x else 0
    y = rng.randint(0, max_y) if max_y else 0
    canvas.paste(mol_image, (x, y))

    if rng.random() < 0.12:
        draw = ImageDraw.Draw(canvas)
        font = load_font(rng.randint(14, 20))
        text = rng.choice(["not target: NaCl", "C6H12O6 ???", "ref. A-17", "ignore"])
        tx = rng.randint(8, max(8, canvas.width - 180))
        ty = rng.randint(8, max(8, canvas.height - 32))
        draw.text((tx, ty), text, fill=(150, 150, 150), font=font)

    return canvas, [x, y, x + mol_image.width, y + mol_image.height]


def make_sft_row(raw_row: dict, image_prefix: str) -> dict:
    block = raw_row["layout_blocks"][0]
    target = json.dumps({"blocks": [block]}, ensure_ascii=False, separators=(",", ":"))
    return {
        "messages": [
            {"role": "user", "content": "<image>Chemical Structure Recognition:"},
            {"role": "assistant", "content": target},
        ],
        "images": [str(Path(image_prefix) / Path(raw_row["image_path"]).name)],
        "meta": {
            "id": raw_row["id"],
            "variant": "structure",
            "formula": raw_row["formula"],
            "canonical_smiles": raw_row["canonical_smiles"],
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_split(
    split: str,
    records: list[MoleculeRecord],
    output_dir: Path,
    seed: int,
    start_index: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    image_dir = output_dir / "sft" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    sft_rows = []
    for offset, record in enumerate(records):
        index = start_index + offset
        image, bbox = compose_structure_image(record, rng)
        item_id = f"{split}_{index:06d}"
        image_path = image_dir / f"{item_id}.png"
        image.save(image_path, optimize=True)
        block = {
            "type": "chemical_structure",
            "text": "",
            "bbox": bbox,
            "formula": record.formula,
            "elemental_counts": record.elemental_counts,
            "smiles": record.canonical_smiles,
            "name": f"Synthetic {index:06d}",
            "name_cn": f"Synthetic {index:06d}",
        }
        raw = {
            "id": item_id,
            "variant": "structure",
            "layout": "single_formula_curriculum",
            "image_path": str(image_path),
            "svg_path": "",
            "name": f"Synthetic {index:06d}",
            "name_cn": f"Synthetic {index:06d}",
            "formula": record.formula,
            "elemental_counts": record.elemental_counts,
            "canonical_smiles": record.canonical_smiles,
            "molecular_weight": record.molecular_weight,
            "ocr_text": "",
            "layout_blocks": [block],
            "targets": {
                "formula": record.formula,
                "elemental_counts": record.elemental_counts,
                "canonical_smiles": record.canonical_smiles,
                "layout_blocks": [block],
            },
        }
        raw_rows.append(raw)
        sft_rows.append(make_sft_row(raw, "images"))
        if (offset + 1) % 1000 == 0:
            print(f"rendered {split}: {offset + 1}/{len(records)}", flush=True)
    return raw_rows, sft_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strict-split single-molecule formula curriculum data.")
    parser.add_argument("--output-dir", default="output/formula_curriculum_v1")
    parser.add_argument("--train-count", type=int, default=16000)
    parser.add_argument("--eval-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    total = args.train_count + args.eval_count + args.test_count
    records = generate_records(total, args.seed)
    train_records = records[: args.train_count]
    eval_records = records[args.train_count : args.train_count + args.eval_count]
    test_records = records[args.train_count + args.eval_count :]

    split_specs = [
        ("train", train_records, 1),
        ("eval", eval_records, args.train_count + 1),
        ("test", test_records, args.train_count + args.eval_count + 1),
    ]
    summary = {
        "train": len(train_records),
        "eval": len(eval_records),
        "test": len(test_records),
        "seed": args.seed,
        "split_policy": "canonical_smiles are generated once, shuffled, then split without overlap",
        "target": "single chemical_structure block with formula, elemental_counts, and canonical SMILES",
    }
    all_canonical = {}
    for split, split_records, start_index in split_specs:
        raw_rows, sft_rows = write_split(split, split_records, output_dir, args.seed + start_index, start_index)
        write_jsonl(output_dir / "raw" / f"{split}_labels.jsonl", raw_rows)
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", sft_rows)
        if split == "train":
            write_jsonl(output_dir / "sft" / "smoke_train.jsonl", sft_rows[:8])
        elif split == "eval":
            write_jsonl(output_dir / "sft" / "smoke_eval.jsonl", sft_rows[:8])
        all_canonical[split] = {record.canonical_smiles for record in split_records}

    summary["overlap"] = {
        "train_eval": len(all_canonical["train"] & all_canonical["eval"]),
        "train_test": len(all_canonical["train"] & all_canonical["test"]),
        "eval_test": len(all_canonical["eval"] & all_canonical["test"]),
    }
    summary["files"] = {
        "train": str(output_dir / "sft" / "train.jsonl"),
        "eval": str(output_dir / "sft" / "eval.jsonl"),
        "test": str(output_dir / "sft" / "test.jsonl"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

