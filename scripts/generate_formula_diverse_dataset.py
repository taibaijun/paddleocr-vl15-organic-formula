import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


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
    ("C", 58),
    ("O", 15),
    ("N", 11),
    ("S", 4),
    ("F", 4),
    ("Cl", 5),
    ("Br", 3),
]

LINEAR_ALKYLS = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"]
ALKYLS = LINEAR_ALKYLS + ["CC(C)", "CC(C)C", "CCC(C)", "C(C)(C)C"]
HALOGENS = ["F", "Cl", "Br"]
PREFIX_SUBS = [
    "C",
    "CC",
    "CCC",
    "CCCC",
    "CC(C)",
    "O",
    "OC",
    "OCC",
    "N",
    "NC",
    "N(C)C",
    "S",
    "SC",
    "F",
    "Cl",
    "Br",
    "N#C",
    "O=C(O)",
    "O=C(N)",
    "CC(=O)",
    "COC(=O)",
]
BRANCH_SUBS = [
    "C",
    "CC",
    "CCC",
    "CCCC",
    "CC(C)",
    "O",
    "OC",
    "OCC",
    "N",
    "NC",
    "N(C)C",
    "S",
    "SC",
    "F",
    "Cl",
    "Br",
    "C#N",
    "C(=O)O",
    "C(=O)N",
    "C(=O)C",
    "C(=O)OC",
    "OC(=O)C",
]


@dataclass(frozen=True)
class MoleculeRecord:
    canonical_smiles: str
    formula: str
    elemental_counts: dict[str, int]
    molecular_weight: float
    atom_count: int
    family: str


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
    if parent_symbol == "C" and child_symbol in {"C", "O", "N", "S"} and parent_remaining >= 2 and max_child >= 2:
        possible.append(2)
    if parent_symbol == "C" and child_symbol in {"C", "N"} and parent_remaining >= 3 and max_child >= 3:
        possible.append(3)

    weights = []
    for order in possible:
        if order == 1:
            weights.append((order, 82))
        elif order == 2:
            weights.append((order, 14))
        else:
            weights.append((order, 4))

    total = sum(weight for _, weight in weights)
    pick = rng.randint(1, total)
    upto = 0
    for order, weight in weights:
        upto += weight
        if pick <= upto:
            if order > parent_remaining or order > max_child:
                return None
            return {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}[order]
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


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def random_alkyl(rng: random.Random, max_len: int = 10) -> str:
    if rng.random() < 0.62:
        return "C" * rng.randint(1, max_len)
    left = "C" * rng.randint(0, max(0, max_len // 3))
    main = "C" * rng.randint(1, max(1, max_len // 2))
    branch = "C" * rng.randint(1, max(1, max_len // 3))
    right = "C" * rng.randint(0, max(0, max_len // 3))
    return f"{left}{main}({branch}){right}"


def random_aliphatic_smiles(rng: random.Random) -> str:
    r1 = random_alkyl(rng)
    r2 = random_alkyl(rng)
    r3 = random_alkyl(rng, max_len=5)
    templates = [
        f"{r1}O",
        f"{r1}N",
        f"{r1}S",
        f"{r1}{rng.choice(HALOGENS)}",
        f"{r1}C#N",
        f"{r1}C(=O)O",
        f"{r1}C(=O)N",
        f"{r1}C(=O)",
        f"{r1}O{r2}",
        f"{r1}S{r2}",
        f"{r1}C(=O){r2}",
        f"{r1}OC(=O){r2}",
        f"{r1}C(=O)O{r2}",
        f"{r1}C(=O)N{r2}",
        f"{r1}N({r2}){r3}",
    ]
    n1 = rng.randint(1, 4)
    n2 = rng.randint(1, 4)
    templates.extend(
        [
            "C" * n1 + "=C" + "C" * n2,
            "C" * n1 + "#C" + "C" * n2,
            "C" * n1 + "C(=O)OC" + "C" * n2,
        ]
    )
    return rng.choice(templates)


def random_ring_smiles(rng: random.Random) -> str:
    r = random_alkyl(rng, max_len=6)
    s1 = rng.choice(BRANCH_SUBS)
    s2 = rng.choice(BRANCH_SUBS)
    s3 = rng.choice(BRANCH_SUBS)
    hal = rng.choice(HALOGENS)
    templates = [
        "C1CCCCC1",
        "C1CCCC1",
        "C1CCCCC1O",
        "O=C1CCCCC1",
        "C1=CCCCC1",
        "C1CCOC1",
        "C1COCCO1",
        "C1CCNCC1",
        "C1CCSCC1",
        f"{r}C1CCCCC1",
        f"{r}OC1CCCCC1",
        f"{hal}C1CCCCC1",
        f"{r}C1=CCCCC1",
        f"{r}C(=O)C1CCCCC1",
        f"C1({s1})CCCCC1",
        f"C1({s1})CC({s2})CCC1",
        f"C1({s1})CC({s2})CC({s3})C1",
        f"C1({s1})CCC({s2})C1",
        f"C1({s1})CCOC({s2})1",
        f"C1({s1})CCN({r})CC1",
        f"O=C1CC({s1})CCC1",
    ]
    return rng.choice(templates)


def random_aromatic_smiles(rng: random.Random) -> str:
    r = random_alkyl(rng, max_len=6)
    r2 = random_alkyl(rng, max_len=4)
    hal = rng.choice(HALOGENS)
    p1 = rng.choice(PREFIX_SUBS)
    p2 = rng.choice(PREFIX_SUBS)
    s1 = rng.choice(BRANCH_SUBS)
    s2 = rng.choice(BRANCH_SUBS)
    s3 = rng.choice(BRANCH_SUBS)
    templates = [
        "c1ccccc1",
        f"{r}c1ccccc1",
        f"{hal}c1ccccc1",
        "Oc1ccccc1",
        "Nc1ccccc1",
        f"{r}Oc1ccccc1",
        f"{r}Nc1ccccc1",
        "O=C(O)c1ccccc1",
        "O=Cc1ccccc1",
        "N#Cc1ccccc1",
        f"{r}C(=O)c1ccccc1",
        f"{r}OC(=O)c1ccccc1",
        f"{p1}c1c({s1})cccc1",
        f"{p1}c1cc({s1})ccc1",
        f"{p1}c1ccc({s1})cc1",
        f"{p1}c1c({s1})cc({s2})cc1",
        f"{p1}c1cc({s1})c({s2})cc1",
        f"{p1}c1c({s1})c({s2})ccc1",
        f"{p1}c1c({s1})c({s2})c({s3})cc1",
        f"{p1}c1cc({s1})c({s2})c({s3})c1",
        f"{p1}c1ccc({s1})c({s2})c1",
        f"{r}c1ccc(O)cc1",
        f"{hal}c1ccc({r2})cc1",
        f"{r}C(=O)Nc1ccccc1",
        "c1ccncc1",
        f"{p2}c1ccncc1",
        f"{hal}c1ccncc1",
        "Oc1ccncc1",
        f"{p2}c1cc({s1})ncc1",
        f"{p2}c1ccncc1{s1}",
        "c1ccoc1",
        f"{p2}c1ccoc1",
        f"{hal}c1ccoc1",
        f"{p2}c1c({s1})coc1",
        "c1ccsc1",
        f"{p2}c1ccsc1",
        f"{hal}c1ccsc1",
        f"{p2}c1c({s1})csc1",
    ]
    return rng.choice(templates)


def sampled_molecule_for_family(rng: random.Random, family: str) -> Chem.Mol | None:
    if family == "random_tree":
        return random_tree_mol(rng, rng.randint(4, 20))
    if family == "aliphatic_functional":
        return mol_from_smiles(random_aliphatic_smiles(rng))
    if family == "ring_alicyclic":
        return mol_from_smiles(random_ring_smiles(rng))
    return mol_from_smiles(random_aromatic_smiles(rng))


def choose_family_with_quota(rng: random.Random, quotas: dict[str, int], counts: Counter[str]) -> str:
    remaining = [(family, max(0, quota - counts[family])) for family, quota in quotas.items()]
    remaining = [(family, count) for family, count in remaining if count > 0]
    if not remaining:
        return "random_tree"
    total = sum(count for _, count in remaining)
    pick = rng.randint(1, total)
    upto = 0
    for family, count in remaining:
        upto += count
        if pick <= upto:
            return family
    return remaining[-1][0]


def make_record(mol: Chem.Mol, family: str) -> MoleculeRecord | None:
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
    if atom_count < 3 or atom_count > 26 or molecular_weight < 35 or molecular_weight > 420:
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
        family=family,
    )


def generate_records(total: int, seed: int) -> list[MoleculeRecord]:
    rng = random.Random(seed)
    records: dict[str, MoleculeRecord] = {}
    family_counts: Counter[str] = Counter()
    quotas = {
        "random_tree": int(total * 0.43),
        "aliphatic_functional": int(total * 0.20),
        "ring_alicyclic": int(total * 0.12),
        "aromatic_heteroaromatic": total - int(total * 0.43) - int(total * 0.20) - int(total * 0.12),
    }
    attempts = 0
    max_attempts = max(120000, total * 300)
    while len(records) < total and attempts < max_attempts:
        attempts += 1
        family = choose_family_with_quota(rng, quotas, family_counts)
        mol = sampled_molecule_for_family(rng, family)
        if mol is None:
            continue
        record = make_record(mol, family)
        if record is None or record.canonical_smiles in records:
            continue
        if family_counts[record.family] >= quotas[record.family]:
            continue
        records[record.canonical_smiles] = record
        family_counts[record.family] += 1
        if len(records) % 1000 == 0:
            print(f"unique molecules: {len(records)}/{total} {dict(family_counts)}", flush=True)
    if len(records) < total:
        raise RuntimeError(f"Generated only {len(records)} unique molecules after {attempts} attempts.")
    values = list(records.values())
    rng.shuffle(values)
    return values


def draw_mol_png(mol: Chem.Mol, size: tuple[int, int], rng: random.Random) -> Image.Image:
    AllChem.Compute2DCoords(mol)
    drawer = Draw.MolDraw2DCairo(size[0], size[1])
    options = drawer.drawOptions()
    options.padding = rng.uniform(0.04, 0.17)
    options.bondLineWidth = rng.uniform(1.4, 3.3)
    options.fixedBondLength = rng.uniform(23, 46)
    options.rotate = rng.choice([0, 0, 0, 15, 345, 90, 180, 270])
    if rng.random() < 0.18:
        options.useBWAtomPalette()
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    image = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")
    if rng.random() < 0.22:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.08, 0.36)))
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


def draw_noise(canvas: Image.Image, rng: random.Random) -> None:
    draw = ImageDraw.Draw(canvas)
    if rng.random() < 0.32:
        for _ in range(rng.randint(1, 4)):
            y = rng.randint(12, canvas.height - 12)
            draw.line((rng.randint(0, 40), y, canvas.width - rng.randint(0, 40), y), fill=(rng.randint(218, 238),) * 3, width=1)
    if rng.random() < 0.3:
        for _ in range(rng.randint(1, 3)):
            x0 = rng.randint(5, canvas.width - 120)
            y0 = rng.randint(5, canvas.height - 90)
            x1 = min(canvas.width - 5, x0 + rng.randint(80, 260))
            y1 = min(canvas.height - 5, y0 + rng.randint(40, 160))
            draw.rectangle((x0, y0, x1, y1), outline=(rng.randint(212, 234),) * 3, width=1)
    if rng.random() < 0.2:
        font = load_font(rng.randint(12, 18))
        distractors = [
            "not target: NaCl",
            "ignore C6H12O6",
            "R1/R2 table",
            "sample 7A",
            "yield 84%",
            "solvent DMSO",
        ]
        for _ in range(rng.randint(1, 3)):
            draw.text(
                (rng.randint(8, max(8, canvas.width - 180)), rng.randint(8, max(8, canvas.height - 28))),
                rng.choice(distractors),
                fill=(rng.randint(130, 175),) * 3,
                font=font,
            )


def compose_structure_image(record: MoleculeRecord, rng: random.Random) -> tuple[Image.Image, list[int]]:
    mol = Chem.MolFromSmiles(record.canonical_smiles)
    if mol is None:
        raise ValueError(record.canonical_smiles)

    canvas_size = rng.choice([(720, 520), (760, 560), (820, 620), (900, 640), (960, 720), (1024, 720)])
    mol_size = (
        rng.randint(390, min(820, canvas_size[0] - 30)),
        rng.randint(280, min(580, canvas_size[1] - 30)),
    )
    mol_image = draw_mol_png(mol, mol_size, rng)
    bg_value = rng.randint(247, 255)
    canvas = Image.new("RGB", canvas_size, (bg_value, bg_value, bg_value))
    draw_noise(canvas, rng)
    max_x = max(0, canvas.width - mol_image.width)
    max_y = max(0, canvas.height - mol_image.height)
    x = rng.randint(0, max_x) if max_x else 0
    y = rng.randint(0, max_y) if max_y else 0
    canvas.paste(mol_image, (x, y))
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
            "variant": "diverse_structure",
            "formula": raw_row["formula"],
            "canonical_smiles": raw_row["canonical_smiles"],
            "family": raw_row["family"],
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
) -> tuple[list[dict], list[dict], Counter]:
    rng = random.Random(seed)
    image_dir = output_dir / "sft" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    sft_rows = []
    families: Counter[str] = Counter()
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
            "name": f"Diverse {index:06d}",
            "name_cn": f"Diverse {index:06d}",
        }
        raw = {
            "id": item_id,
            "variant": "diverse_structure",
            "layout": "single_formula_diverse",
            "family": record.family,
            "image_path": str(image_path),
            "svg_path": "",
            "name": f"Diverse {index:06d}",
            "name_cn": f"Diverse {index:06d}",
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
        families[record.family] += 1
        raw_rows.append(raw)
        sft_rows.append(make_sft_row(raw, "images"))
        if (offset + 1) % 1000 == 0:
            print(f"rendered {split}: {offset + 1}/{len(records)}", flush=True)
    return raw_rows, sft_rows, families


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diverse organic molecule formula data with rings, aromatics and functional groups.")
    parser.add_argument("--output-dir", default="output/formula_diverse_v6")
    parser.add_argument("--train-count", type=int, default=20000)
    parser.add_argument("--eval-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260510)
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
        "target": "single chemical_structure block with formula, elemental_counts and canonical SMILES",
        "family_distribution": {},
    }
    all_canonical = {}
    for split, split_records, start_index in split_specs:
        raw_rows, sft_rows, families = write_split(split, split_records, output_dir, args.seed + start_index, start_index)
        write_jsonl(output_dir / "raw" / f"{split}_labels.jsonl", raw_rows)
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", sft_rows)
        if split == "train":
            write_jsonl(output_dir / "sft" / "smoke_train.jsonl", sft_rows[:8])
        elif split == "eval":
            write_jsonl(output_dir / "sft" / "smoke_eval.jsonl", sft_rows[:8])
        all_canonical[split] = {record.canonical_smiles for record in split_records}
        summary["family_distribution"][split] = dict(sorted(families.items()))

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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

