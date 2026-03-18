"""
predict_report.py

Retorna o laudo JSON associado a uma imagem, usando o mesmo formato de treino.
- Busca pelo nome do arquivo OU pelo relations map (image_report_map.json).
- Normaliza o JSON para o schema "Dental OCR FDI Response".

Uso:
    python -m inference.predict_report --image datasets/raw/images/PAN\ 01.jpg
"""

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "datasets" / "raw" / "reports"
DEFAULT_RELATIONS_PATH = PROJECT_ROOT / "datasets" / "raw" / "relations" / "image_report_map.json"
DEFAULT_INFERENCE_CONFIG = PROJECT_ROOT / "configs" / "inference.yaml"
DEFAULT_FDI_CLASS_MAP = PROJECT_ROOT / "configs" / "fdi_class_map.json"

FDI_TEETH = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
    "51", "52", "53", "54", "55", "61", "62", "63", "64", "65",
    "71", "72", "73", "74", "75", "81", "82", "83", "84", "85",
]


def _empty_findings():
    return {
        "caries": None,
        "restoration": None,
        "restoration_material": None,
        "endodontic_treatment": None,
        "crown": None,
        "implant": None,
        "fracture": None,
        "mobility": None,
        "bone_loss": None,
        "periodontal_involvement": None,
        "periapical_lesion": None,
        "resorption_internal": None,
        "resorption_external": None,
        "calcification": None,
        "hypoplasia": None,
        "supernumerary_relation": None,
        "notes": None,
    }


def _empty_tooth(fdi):
    return {
        "fdi": fdi,
        "status": None,
        "present": None,
        "erupted": None,
        "findings": _empty_findings(),
    }


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _apply_findings_from_text(findings: dict, text: str) -> dict:
    normalized = _normalize_text(text)

    if "cari" in normalized:
        findings["caries"] = True
    if "restaur" in normalized or "restora" in normalized:
        findings["restoration"] = True
    if "endodont" in normalized or "canal" in normalized:
        findings["endodontic_treatment"] = True
    if "coroa" in normalized:
        findings["crown"] = True
    if "impl" in normalized:
        findings["implant"] = True
    if "fratur" in normalized:
        findings["fracture"] = True
    if "mobil" in normalized:
        findings["mobility"] = "present"
    if "perda ossea" in normalized or "perda óssea" in normalized:
        findings["bone_loss"] = "present"
    if "furca" in normalized:
        findings["periodontal_involvement"] = "furcation"
    if "periap" in normalized or "apical" in normalized:
        findings["periapical_lesion"] = True
    if "reabsorc" in normalized or "resorc" in normalized:
        findings["resorption_external"] = True
    if "calcific" in normalized:
        findings["calcification"] = "present"
    if "hipoplas" in normalized:
        findings["hypoplasia"] = "present"
    if "supranumer" in normalized:
        findings["supernumerary_relation"] = "present"

    findings["notes"] = text
    return findings


def _build_tooth_from_description(fdi: str, description: str) -> dict:
    tooth = _empty_tooth(fdi)
    if not description:
        return tooth

    normalized = _normalize_text(description)

    if "missing" in normalized or "ausente" in normalized:
        tooth["status"] = "missing"
        tooth["present"] = False
        tooth["erupted"] = None
        return tooth

    if "normal" in normalized or "higido" in normalized or "hígido" in normalized:
        tooth["status"] = "normal"
        tooth["present"] = True
        tooth["erupted"] = True
        tooth["findings"]["notes"] = description
        return tooth

    tooth["status"] = "unknown"
    tooth["present"] = True
    tooth["erupted"] = None
    tooth["findings"] = _apply_findings_from_text(tooth["findings"], description)
    return tooth


def _normalize_state_dict(state_dict: dict) -> dict:
    """Remove common wrappers/prefixes from checkpoint state_dict keys."""
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        cleaned[key] = value
    return cleaned


def _load_checkpoint_state_dict(checkpoint_path: Path, device):
    """Load checkpoint and return a normalized model state_dict."""
    import torch
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    return _normalize_state_dict(state_dict)


def _predict_mask(image_path: Path, config_path: Path):
    try:
        import cv2
        import numpy as np
        import torch
        import yaml
        from torch.cuda.amp import autocast

        from monai_core.models.segmentation_model import build_segmentation_model
        from monai_core.postprocessing.segmentation_post import get_post_transforms
    except ModuleNotFoundError:
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = PROJECT_ROOT / cfg["segmentation_checkpoint"]
    image_size = cfg["image_size_segmentation"]
    num_classes = cfg.get("num_classes", 2)

    if not checkpoint_path.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmentation_model(num_classes=num_classes).to(device)
    state_dict = _load_checkpoint_state_dict(checkpoint_path, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "⚠️  Aviso: chaves faltantes/inesperadas no checkpoint. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.eval()

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler: {image_path}")

    img_resized = cv2.resize(img, tuple(image_size), interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
    post_pred, _ = get_post_transforms(num_classes=num_classes)
    use_amp = device.type == "cuda"

    with torch.no_grad(), autocast(enabled=use_amp):
        output = model(input_tensor)
        prediction = post_pred(output.float())

    return prediction.squeeze().cpu().numpy().astype(np.uint8)


def _load_class_map(class_map_path: Path) -> dict:
    if not class_map_path.exists():
        return {}
    with open(class_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_report_from_segmentation(mask, class_map: dict) -> dict:
    report = build_empty_report()
    report["exam_context"]["source_type"] = "model_inference"

    if class_map:
        import numpy as np
        unique_labels = set(int(v) for v in np.unique(mask))
        fdi_labels = {int(v) for v in class_map.values()}
        if unique_labels & fdi_labels:
            present_teeth = []
            for fdi, class_id in class_map.items():
                tooth = report["odontogram"].get(fdi) or _empty_tooth(fdi)
                if int(class_id) in unique_labels:
                    tooth["status"] = "normal"
                    tooth["present"] = True
                    tooth["erupted"] = True
                    present_teeth.append(fdi)
                else:
                    tooth["status"] = "missing"
                    tooth["present"] = False
                    tooth["erupted"] = None
                report["odontogram"][fdi] = tooth

            report["impression"]["summary"] = [
                f"Detected teeth: {', '.join(sorted(present_teeth)) if present_teeth else 'none'}"
            ]
            report["impression"]["review_status"] = "model_inference_fdi"
            return report

    tooth_pixels = int((mask == 1).sum())
    caries_pixels = int((mask == 2).sum())

    if tooth_pixels > 0:
        for fdi in FDI_TEETH:
            tooth = report["odontogram"][fdi]
            tooth["status"] = "normal"
            tooth["present"] = True
            tooth["erupted"] = True
            report["odontogram"][fdi] = tooth

    if caries_pixels > 0:
        ratio = min(caries_pixels / max(tooth_pixels, 1), 1.0)
        report["global_findings"]["notes"] = "caries detected (not localized)"
        report["impression"]["summary"] = [
            "Caries detected in segmentation mask (localization not available)."
        ]
        report["impression"]["confidence"] = round(ratio, 3)
        report["impression"]["review_status"] = "model_inference_unlocalized"

        for fdi in FDI_TEETH:
            tooth = report["odontogram"][fdi]
            tooth["findings"]["caries"] = True
            tooth["findings"]["notes"] = "caries detected (not localized)"
            report["odontogram"][fdi] = tooth

    return report


def build_empty_report():
    return {
        "exam_type": "panoramic_radiograph",
        "patient": {
            "patient_id": None,
            "exam_id": None,
            "exam_date": None,
            "sex": None,
            "birth_date": None,
        },
        "exam_context": {
            "source_type": None,
            "modality": None,
            "report_language": None,
        },
        "odontogram": {fdi: _empty_tooth(fdi) for fdi in FDI_TEETH},
        "global_findings": {
            "mixed_dentition": None,
            "malocclusion": None,
            "crowding": None,
            "spacing": None,
            "crossbite": None,
            "open_bite": None,
            "deep_bite": None,
            "overjet": None,
            "overbite": None,
            "midline_deviation": None,
            "edentulism": None,
            "prostheses": None,
            "orthodontic_appliance": None,
            "bone_pattern": None,
            "tmj_findings": None,
            "sinus_findings": None,
            "soft_tissue_findings": None,
            "notes": None,
        },
        "impression": {
            "summary": ["report_placeholder_generated"],
            "confidence": None,
            "review_status": None,
        },
    }


def _load_relations(relations_path: Path) -> list:
    if not relations_path.exists():
        return []
    with open(relations_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_report_path(image_path: Path, reports_dir: Path, relations_path: Path) -> Path:
    direct_path = reports_dir / f"{image_path.stem}.json"
    if direct_path.exists():
        return direct_path

    image_name = image_path.name
    for entry in _load_relations(relations_path):
        if entry.get("image_file") == image_name:
            candidate = reports_dir / entry.get("report_file", "")
            if candidate.exists():
                return candidate
    return direct_path


def normalize_report(raw_report: dict) -> dict:
    """Normaliza o JSON do laudo para o schema esperado (dental OCR FDI response)."""
    report = build_empty_report()

    dental_report = raw_report.get("dentalReport", {}) if isinstance(raw_report, dict) else {}
    if not isinstance(dental_report, dict):
        return report

    report["exam_type"] = dental_report.get("exam_type") or report["exam_type"]
    report["patient"] = {
        "patient_id": dental_report.get("patient", {}).get("patient_id"),
        "exam_id": dental_report.get("patient", {}).get("exam_id"),
        "exam_date": dental_report.get("patient", {}).get("exam_date"),
        "sex": dental_report.get("patient", {}).get("sex"),
        "birth_date": dental_report.get("patient", {}).get("birth_date"),
    }
    report["exam_context"] = {
        "source_type": dental_report.get("exam_context", {}).get("source_type"),
        "modality": dental_report.get("exam_context", {}).get("modality"),
        "report_language": dental_report.get("exam_context", {}).get("report_language"),
    }
    report["global_findings"] = {
        "mixed_dentition": dental_report.get("global_findings", {}).get("mixed_dentition"),
        "malocclusion": dental_report.get("global_findings", {}).get("malocclusion"),
        "crowding": dental_report.get("global_findings", {}).get("crowding"),
        "spacing": dental_report.get("global_findings", {}).get("spacing"),
        "crossbite": dental_report.get("global_findings", {}).get("crossbite"),
        "open_bite": dental_report.get("global_findings", {}).get("open_bite"),
        "deep_bite": dental_report.get("global_findings", {}).get("deep_bite"),
        "overjet": dental_report.get("global_findings", {}).get("overjet"),
        "overbite": dental_report.get("global_findings", {}).get("overbite"),
        "midline_deviation": dental_report.get("global_findings", {}).get("midline_deviation"),
        "edentulism": dental_report.get("global_findings", {}).get("edentulism"),
        "prostheses": dental_report.get("global_findings", {}).get("prostheses"),
        "orthodontic_appliance": dental_report.get("global_findings", {}).get("orthodontic_appliance"),
        "bone_pattern": dental_report.get("global_findings", {}).get("bone_pattern"),
        "tmj_findings": dental_report.get("global_findings", {}).get("tmj_findings"),
        "sinus_findings": dental_report.get("global_findings", {}).get("sinus_findings"),
        "soft_tissue_findings": dental_report.get("global_findings", {}).get("soft_tissue_findings"),
        "notes": dental_report.get("global_findings", {}).get("notes"),
    }
    report["impression"] = {
        "summary": dental_report.get("impression", {}).get("summary", []),
        "confidence": dental_report.get("impression", {}).get("confidence"),
        "review_status": dental_report.get("impression", {}).get("review_status"),
    }

    odontogram = dental_report.get("odontogram", {})
    if isinstance(odontogram, dict):
        for fdi in FDI_TEETH:
            report["odontogram"][fdi] = _build_tooth_from_description(
                fdi,
                odontogram.get(fdi),
            )

    return report


def load_or_build_report(
    image_path: Path,
    reports_dir: Path,
    relations_path: Path,
    config_path: Path,
    class_map_path: Path,
) -> dict:
    report_path = find_report_path(image_path, reports_dir, relations_path)
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            raw_report = json.load(f)
        return normalize_report(raw_report)
    mask = _predict_mask(image_path, config_path)
    if mask is None:
        report = build_empty_report()
        report["impression"]["summary"] = ["no_report_and_no_checkpoint"]
        return report
    class_map = _load_class_map(class_map_path)
    return _build_report_from_segmentation(mask, class_map)


def main():
    parser = argparse.ArgumentParser(description="Retorna JSON do laudo conforme treinamento")
    parser.add_argument("--image", required=True, help="Caminho da imagem")
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Diretório de laudos (JSON) com mesmo nome da imagem",
    )
    parser.add_argument(
        "--relations",
        default=str(DEFAULT_RELATIONS_PATH),
        help="Arquivo JSON de relações imagem→laudo",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_INFERENCE_CONFIG),
        help="Config YAML de inferência (checkpoint + image_size)",
    )
    parser.add_argument(
        "--class-map",
        default=str(DEFAULT_FDI_CLASS_MAP),
        help="Mapeamento FDI -> classe para segmentação por dente",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    reports_dir = Path(args.reports_dir)
    relations_path = Path(args.relations)
    config_path = Path(args.config)
    class_map_path = Path(args.class_map)

    report = load_or_build_report(
        image_path,
        reports_dir,
        relations_path,
        config_path,
        class_map_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
