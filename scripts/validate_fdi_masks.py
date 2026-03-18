"""
validate_fdi_masks.py

Valida se as máscaras FDI possuem apenas classes permitidas.

Uso:
    python scripts/validate_fdi_masks.py --masks datasets/processed/masks --class-map configs/fdi_class_map.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_class_map(path: Path) -> set:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(v) for v in data.values()}


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Validate FDI masks")
    parser.add_argument("--masks", required=True, help="Diretório de máscaras")
    parser.add_argument("--class-map", required=True, help="Arquivo JSON FDI -> classe")
    args = parser.parse_args()

    mask_dir = Path(args.masks)
    class_labels = load_class_map(Path(args.class_map))
    allowed = class_labels | {0}

    masks = sorted(mask_dir.glob("*.png"))
    if not masks:
        print("⚠ Nenhuma máscara encontrada para validar.")
        return

    invalid = []
    for mask_path in masks:
        mask = read_mask(mask_path)
        labels = set(int(v) for v in np.unique(mask))
        diff = labels - allowed
        if diff:
            invalid.append((mask_path.name, sorted(diff)))

    if invalid:
        print("❌ Máscaras com classes inválidas:")
        for name, diff in invalid:
            print(f"  {name}: {diff}")
        raise SystemExit(1)

    print(f"✅ {len(masks)} máscaras OK. Classes permitidas: {sorted(allowed)}")


if __name__ == "__main__":
    main()
