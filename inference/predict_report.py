"""
predict_report.py

Retorna o laudo JSON associado a uma imagem, usando o mesmo formato de treino.
- Se existir um laudo com o mesmo nome (stem) em datasets/raw/reports, retorna esse JSON.
- Caso contrário, gera um JSON vazio seguindo o schema esperado.

Uso:
    python -m inference.predict_report --image datasets/raw/images/PAN\ 01.jpg
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "datasets" / "raw" / "reports"

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


def find_report_path(image_path: Path, reports_dir: Path) -> Path:
    stem = image_path.stem
    return reports_dir / f"{stem}.json"


def load_or_build_report(image_path: Path, reports_dir: Path) -> dict:
    report_path = find_report_path(image_path, reports_dir)
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return build_empty_report()


def main():
    parser = argparse.ArgumentParser(description="Retorna JSON do laudo conforme treinamento")
    parser.add_argument("--image", required=True, help="Caminho da imagem")
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Diretório de laudos (JSON) com mesmo nome da imagem",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    reports_dir = Path(args.reports_dir)

    report = load_or_build_report(image_path, reports_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
