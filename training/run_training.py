"""
run_training.py

Pipeline completo de treinamento de segmentação com MONAI.
Otimizado para RunPod GPUs (AMP, cuDNN benchmark, torch.compile).
"""

import os
import sys
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch.amp import autocast, GradScaler

# Adiciona o root do projeto ao path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from monai.data import DataLoader, Dataset
from monai.transforms import Compose, AsDiscrete, Activations
from monai.metrics import DiceMetric

from monai_core.transforms.train_transforms import get_train_transforms, get_val_transforms
from monai_core.models.segmentation_model import build_segmentation_model
from monai_core.losses.dice_loss import get_loss_function

import json


def _get_cfg(config, key, default=None):
    """Busca chave suportando config flat ou aninhada."""
    return config.get(key, default)


def _get_group(config, group_name, defaults=None):
    """Retorna grupo aninhado se existir, senão usa config flat."""
    if group_name in config and isinstance(config[group_name], dict):
        return config[group_name]
    return defaults or {}


def print_gpu_info():
    """Mostra informações da GPU."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / (1024 ** 3)
            print(f"  GPU {i}: {props.name} ({vram:.1f} GB VRAM)")
    else:
        print("  ⚠ Nenhuma GPU CUDA detectada")


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, use_amp=True):
    """Treina uma época."""
    model.train()
    epoch_loss = 0
    scaler = GradScaler('cuda', enabled=use_amp)

    for batch_idx, batch_data in enumerate(loader):
        images = batch_data["image"].to(device, non_blocking=True)
        labels = batch_data["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        if batch_idx % 5 == 0:
            print(f"    Batch {batch_idx}/{len(loader)} — Loss: {loss.item():.4f}")

    return epoch_loss / max(len(loader), 1)


def validate(model, loader, loss_fn, device, num_classes, use_amp=True):
    """Valida o modelo e retorna loss média e Dice score."""
    model.eval()
    epoch_loss = 0

    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, threshold=0.5)])
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    with torch.no_grad():
        for batch_data in loader:
            images = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True).long()

            with autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            epoch_loss += loss.item()

            outputs_post = [post_pred(o) for o in outputs]
            outputs_onehot = [
                F.one_hot(o.squeeze(0).long(), num_classes).permute(2, 0, 1).unsqueeze(0).float()
                for o in outputs_post
            ]
            labels_onehot = [
                F.one_hot(l.squeeze(0).long(), num_classes).permute(2, 0, 1).unsqueeze(0).float()
                for l in labels
            ]

            outputs_onehot = torch.cat(outputs_onehot, dim=0).to(device)
            labels_onehot = torch.cat(labels_onehot, dim=0).to(device)

            dice_metric(y_pred=outputs_onehot, y=labels_onehot)

    mean_loss = epoch_loss / max(len(loader), 1)
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()

    return mean_loss, dice_score


def load_manifest(manifest_path):
    """Carrega um manifest JSON e retorna lista de dicts com paths absolutos."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    data_list = []
    for entry in entries:
        image_path = str(PROJECT_ROOT / entry["image"])
        label_path = str(PROJECT_ROOT / entry["label"])
        data_list.append({"image": image_path, "label": label_path})

    return data_list


def main():
    # ── Config ─────────────────────────────────────────
    config_path = PROJECT_ROOT / "configs" / "train_segmentation.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # suporta YAML aninhado e flat
    train_cfg = _get_group(config, "training")
    data_cfg = _get_group(config, "data")
    model_cfg = _get_group(config, "model")

    # valores fallback (flat)
    model_cfg.setdefault("num_classes", _get_cfg(config, "num_classes", 2))
    model_cfg.setdefault("in_channels", _get_cfg(config, "in_channels", 1))

    data_cfg.setdefault("train_manifest", _get_cfg(config, "train_manifest"))
    data_cfg.setdefault("val_manifest", _get_cfg(config, "val_manifest"))
    data_cfg.setdefault("image_size", _get_cfg(config, "image_size", [512, 512]))
    data_cfg.setdefault("num_workers", _get_cfg(config, "num_workers", 4))

    train_cfg.setdefault("batch_size", _get_cfg(config, "batch_size", 8))
    train_cfg.setdefault("epochs", _get_cfg(config, "epochs", 100))
    train_cfg.setdefault("learning_rate", _get_cfg(config, "learning_rate", 1e-3))
    train_cfg.setdefault("amp", _get_cfg(config, "amp", True))
    train_cfg.setdefault("checkpoint_name", _get_cfg(config, "checkpoint_name", "best_model.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"

    print("=" * 60)
    print("  RadianceAI MONAI — Treinamento de Segmentação")
    print("=" * 60)
    print(f"  Device: {device}")
    print_gpu_info()
    print(f"  AMP: {use_amp}")
    print(f"  Config: {config_path.name}")
    print("=" * 60)

    # ── GPU otimizações ────────────────────────────────
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Dados ──────────────────────────────────────────
    image_size = tuple(data_cfg.get("image_size", [512, 512]))
    train_manifest = PROJECT_ROOT / data_cfg["train_manifest"]
    val_manifest = PROJECT_ROOT / data_cfg["val_manifest"]

    train_data = load_manifest(train_manifest)
    val_data = load_manifest(val_manifest)

    print(f"\n📂 Train: {len(train_data)} casos")
    print(f"📂 Val:   {len(val_data)} casos")

    train_transforms = get_train_transforms(image_size=image_size)
    val_transforms = get_val_transforms(image_size=image_size)

    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=val_transforms)

    num_workers = data_cfg.get("num_workers", 4)
    batch_size = train_cfg.get("batch_size", 8)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # ── Modelo ─────────────────────────────────────────
    num_classes = model_cfg.get("num_classes", 2)
    in_channels = model_cfg.get("in_channels", 1)

    model = build_segmentation_model(
        num_classes=num_classes,
        in_channels=in_channels,
    ).to(device)

    # Tentar torch.compile (PyTorch 2.x)
    try:
        model = torch.compile(model)
        print("✅ torch.compile ativado")
    except Exception:
        print("⚠ torch.compile não disponível — usando eager mode")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Modelo: DynUNet — {total_params:,} parâmetros")

    # ── Loss / Optimizer / Scheduler ───────────────────
    loss_fn = get_loss_function(num_classes=num_classes)
    lr = train_cfg.get("learning_rate", 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    epochs = train_cfg.get("epochs", 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Checkpoints dir ────────────────────────────────
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──────────────────────────────────
    best_dice = 0.0
    ckpt_name = train_cfg.get("checkpoint_name", "best_model.pt")
    print(f"\n🚀 Iniciando treinamento — {epochs} épocas\n")

    for epoch in range(1, epochs + 1):
        print(f"─── Época {epoch}/{epochs} ───")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, use_amp)

        # Validate
        val_loss, dice_score = validate(
            model, val_loader, loss_fn, device, num_classes, use_amp
        )

        # Scheduler step
        scheduler.step()

        # Log
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {dice_score:.4f} | LR: {current_lr:.6f}")

        # VRAM usage
        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"  VRAM Pico: {peak_mem:.2f} GB")

        # Save best
        if dice_score > best_dice:
            best_dice = dice_score
            ckpt_path = ckpt_dir / ckpt_name
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dice_score": dice_score,
                "config": config,
            }, ckpt_path)
            print(f"  💾 Melhor modelo salvo! Dice: {dice_score:.4f}")

        # Save every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dice_score": dice_score,
                "config": config,
            }, ckpt_path)

        print()

    print("=" * 60)
    print(f"  ✅ Treinamento concluído!")
    print(f"  🏆 Melhor Dice: {best_dice:.4f}")
    print(f"  📂 Checkpoints: {ckpt_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()