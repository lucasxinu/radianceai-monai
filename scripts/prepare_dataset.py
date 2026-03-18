"""
prepare_dataset.py

Lê o image_report_map.json, pré-processa as imagens panorâmicas raw
e gera:
  - datasets/processed/images/   (imagens redimensionadas para treino)
  - datasets/processed/masks/    (máscaras placeholder para anotação)
  - datasets/processed/manifests/train_manifest.json
  - datasets/processed/manifests/val_manifest.json

Uso:
    python scripts/prepare_dataset.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

IMAGES_RAW_DIR = RAW_DIR / "images"
REPORTS_RAW_DIR = RAW_DIR / "reports"
RELATIONS_PATH = RAW_DIR / "relations" / "image_report_map.json"

IMAGES_OUT_DIR = PROCESSED_DIR / "images"
MASKS_OUT_DIR = PROCESSED_DIR / "masks"
LABELS_OUT_DIR = PROCESSED_DIR / "labels"
MANIFESTS_OUT_DIR = PROCESSED_DIR / "manifests"

IMAGE_SIZE = (512, 512)
VAL_SPLIT = 0.2  # 20% para validação


def load_relations():
    """Carrega o mapa de relações image <-> report."""
    with open(RELATIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(src_path, dst_path):
    """Redimensiona a imagem para IMAGE_SIZE e salva como PNG."""
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"  ⚠  Não foi possível ler: {src_path.name}")
        return False
    img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst_path), img_resized)
    return True


def generate_placeholder_mask(dst_path, present_teeth=None, class_map=None):
    """
    Gera uma máscara placeholder.
    - Se class_map/present_teeth forem fornecidos, cria uma máscara multi-classe
      com um grid de células para cada dente presente.
    - Caso contrário, gera máscara zerada (background).

    Salva como grayscale single-channel PNG.
    """
    mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)

    if class_map and present_teeth:
        cols = 8
        rows = int(np.ceil(len(class_map) / cols))
        cell_w = IMAGE_SIZE[0] // cols
        cell_h = IMAGE_SIZE[1] // max(rows, 1)

        for idx, (fdi, class_id) in enumerate(class_map.items()):
            if fdi not in present_teeth:
                continue
            row = idx // cols
            col = idx % cols
            y0 = row * cell_h
            x0 = col * cell_w
            y1 = min(y0 + cell_h, IMAGE_SIZE[1])
            x1 = min(x0 + cell_w, IMAGE_SIZE[0])
            mask[y0:y1, x0:x1] = int(class_id)

    from PIL import Image
    img = Image.fromarray(mask, mode="L")
    img.save(str(dst_path))


def extract_classification_label(report_path):
    """
    Extrai um label de classificação a partir do report (odontograma).
    Retorna um dict com contagens de achados por classe.
    """
    if not report_path.exists() or report_path.stat().st_size == 0:
        return None

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    dental = data.get("dentalReport", {})
    odontogram = dental.get("odontogram", {})

    counts = {
        "missing": 0,
        "caries": 0,
        "restored": 0,
        "endodontic": 0,
        "normal": 0,
        "other": 0,
    }

    keywords_map = {
        "missing": ["missing", "ausente"],
        "caries": ["caries", "cárie", "carie", "recidiva", "recurrent"],
        "restored": ["restored", "restaura"],
        "endodontic": ["endodontic", "endodôntico", "endodontico"],
        "normal": ["normal", "hígido", "higido"],
    }

    for tooth_id, status in odontogram.items():
        if status is None:
            continue
        status_lower = status.lower()
        classified = False
        for label, keywords in keywords_map.items():
            if any(kw in status_lower for kw in keywords):
                counts[label] += 1
                classified = True
        if not classified:
            counts["other"] += 1

    return counts


def extract_present_teeth(report_path, class_map):
    if not report_path.exists() or report_path.stat().st_size == 0:
        return []

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    dental = data.get("dentalReport", {})
    odontogram = dental.get("odontogram", {})

    present = []
    for fdi in class_map.keys():
        status = odontogram.get(fdi)
        if status is None:
            continue
        status_lower = str(status).lower()
        if "missing" in status_lower or "ausente" in status_lower:
            continue
        present.append(fdi)
    return present


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for MONAI")
    parser.add_argument(
        "--fdi-placeholder",
        action="store_true",
        help="Gerar máscaras multi-classe placeholder usando FDI (grid por dente)",
    )
    parser.add_argument(
        "--class-map",
        default=str(PROJECT_ROOT / "configs" / "fdi_class_map.json"),
        help="Mapeamento FDI -> classe",
    )
    args = parser.parse_args()

    class_map = {}
    if args.fdi_placeholder:
        with open(args.class_map, "r", encoding="utf-8") as f:
            class_map = json.load(f)
    print("=" * 60)
    print("  RadianceAI — Prepare Dataset")
    print("=" * 60)

    # Criar diretórios
    for d in [IMAGES_OUT_DIR, MASKS_OUT_DIR, LABELS_OUT_DIR, MANIFESTS_OUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Carregar relations
    relations = load_relations()
    print(f"\n📋 Relations carregadas: {len(relations)} casos")

    manifest_entries = []
    skipped = 0

    for rel in relations:
        study_id = rel["study_id"]
        image_file = rel["image_file"]
        report_file = rel["report_file"]

        src_image = IMAGES_RAW_DIR / image_file
        src_report = REPORTS_RAW_DIR / report_file

        # Nomes de saída (sem espaço, .png)
        safe_name = study_id.lower()  # ex: pan_01
        dst_image = IMAGES_OUT_DIR / f"{safe_name}.png"
        dst_mask = MASKS_OUT_DIR / f"{safe_name}_mask.png"
        dst_label = LABELS_OUT_DIR / f"{safe_name}_label.json"

        print(f"\n🔗 {study_id}: {image_file} ↔ {report_file}")

        # 1. Pré-processar imagem
        if not src_image.exists():
            print(f"  ⚠  Imagem não encontrada: {image_file} — pulando")
            skipped += 1
            continue

        if not preprocess_image(src_image, dst_image):
            skipped += 1
            continue
        print(f"  ✅ Imagem → {dst_image.name}")

        # 2. Gerar máscara placeholder
        if args.fdi_placeholder and class_map:
            present_teeth = extract_present_teeth(src_report, class_map)
            generate_placeholder_mask(dst_mask, present_teeth, class_map)
        else:
            generate_placeholder_mask(dst_mask)
        print(f"  ✅ Máscara placeholder → {dst_mask.name}")

        # 3. Extrair label de classificação do report
        label_data = extract_classification_label(src_report)
        if label_data is not None:
            with open(dst_label, "w", encoding="utf-8") as f:
                json.dump({"study_id": study_id, "findings": label_data}, f, indent=2)
            print(f"  ✅ Label → {dst_label.name}")
        else:
            print(f"  ⚠  Report vazio/inválido → sem label")

        # 4. Adicionar ao manifest (paths relativos à raiz do projeto)
        manifest_entries.append({
            "study_id": study_id,
            "image": f"./datasets/processed/images/{safe_name}.png",
            "label": f"./datasets/processed/masks/{safe_name}_mask.png",
        })

    # Dividir em train / val
    np.random.seed(42)
    indices = np.random.permutation(len(manifest_entries))
    split_idx = int(len(manifest_entries) * (1 - VAL_SPLIT))

    train_entries = [manifest_entries[i] for i in indices[:split_idx]]
    val_entries = [manifest_entries[i] for i in indices[split_idx:]]

    # Salvar manifests
    train_manifest_path = MANIFESTS_OUT_DIR / "train_manifest.json"
    val_manifest_path = MANIFESTS_OUT_DIR / "val_manifest.json"

    with open(train_manifest_path, "w", encoding="utf-8") as f:
        json.dump(train_entries, f, indent=2, ensure_ascii=False)

    with open(val_manifest_path, "w", encoding="utf-8") as f:
        json.dump(val_entries, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"  ✅ Processados: {len(manifest_entries)} | Pulados: {skipped}")
    print(f"  📂 Train: {len(train_entries)} casos → {train_manifest_path.name}")
    print(f"  📂 Val:   {len(val_entries)} casos  → {val_manifest_path.name}")
    if args.fdi_placeholder:
        print(f"  ⚠  Masks FDI são PLACEHOLDER — substitua por anotações reais!")
    else:
        print(f"  ⚠  Masks são PLACEHOLDER — substitua por anotações reais!")
    print("=" * 60)


if __name__ == "__main__":
    main()
