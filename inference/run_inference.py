"""
run_inference.py

Pipeline de inferência: carrega um checkpoint, processa uma imagem
e gera a máscara de segmentação predita.

Uso:
    python -m inference.run_inference --image <caminho_da_imagem>
    python -m inference.run_inference --image datasets/raw/images/PAN\ 01.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast

from monai_core.models.segmentation_model import build_segmentation_model
from monai_core.postprocessing.segmentation_post import get_post_transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "inference.yaml"


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_image(image_path, image_size):
    """Carrega e pré-processa a imagem para inferência."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler: {image_path}")

    img_resized = cv2.resize(img, tuple(image_size), interpolation=cv2.INTER_AREA)
    # Normalizar [0, 1]
    img_norm = img_resized.astype(np.float32) / 255.0
    # Adicionar dimensões: (1, 1, H, W) = (batch, channel, height, width)
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    return tensor


def run_inference(image_path, config_path=None):
    cfg = load_config(config_path or DEFAULT_CONFIG)

    checkpoint_path = PROJECT_ROOT / cfg["segmentation_checkpoint"]
    image_size = cfg["image_size_segmentation"]
    num_classes = cfg.get("num_classes", 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  RadianceAI — Inference Pipeline")
    print("=" * 60)
    print(f"  Image:      {image_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device:     {device}")

    # Carregar modelo
    model = build_segmentation_model(num_classes=num_classes).to(device)

    if not checkpoint_path.exists():
        print(f"\n  ⚠  Checkpoint não encontrado: {checkpoint_path}")
        print(f"  Execute o treinamento primeiro: python -m training.run_training")
        return None

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Modelo carregado (epoch {ckpt.get('epoch', '?')}, dice {ckpt.get('best_dice', '?')})")

    # Pré-processar imagem
    input_tensor = preprocess_image(image_path, image_size).to(device)

    # Inferência com AMP
    post_pred, _ = get_post_transforms(num_classes=num_classes)
    use_amp = device.type == "cuda"

    with torch.no_grad(), autocast(enabled=use_amp):
        output = model(input_tensor)
        prediction = post_pred(output.float())

    # Converter para numpy
    pred_mask = prediction.squeeze().cpu().numpy().astype(np.uint8)

    # Salvar resultado
    output_dir = PROJECT_ROOT / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem.replace(" ", "_").lower()
    output_path = output_dir / f"{image_name}_pred_mask.png"
    cv2.imwrite(str(output_path), pred_mask * 255)

    print(f"\n  ✅ Máscara predita salva: {output_path}")
    print("=" * 60)
    return pred_mask


def main():
    parser = argparse.ArgumentParser(description="RadianceAI — Inferência de Segmentação")
    parser.add_argument("--image", type=str, required=True, help="Caminho da imagem para inferência")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    run_inference(args.image, args.config)


if __name__ == "__main__":
    main()
