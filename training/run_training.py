"""
run_training.py

Loop completo de treinamento de segmentação com MONAI.
Otimizado para GPUs NVIDIA (RunPod) com AMP e cuDNN benchmark.

Uso:
    python -m training.run_training
    python -m training.run_training --config configs/train_segmentation.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from monai_core.dataloaders.manifest_dataset import ManifestDataset
from monai_core.losses.dice_loss import get_loss_function
from monai_core.metrics.dice_metric import get_dice_metric
from monai_core.models.segmentation_model import build_segmentation_model
from monai_core.postprocessing.segmentation_post import get_post_transforms
from monai_core.transforms.train_transforms import get_train_transforms, get_val_transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_segmentation.yaml"


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_gpu_info():
    """Mostra informações da GPU disponível."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_mem / (1024 ** 3)
            print(f"  GPU {i}: {props.name} ({vram:.1f} GB VRAM)")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  cuDNN: {torch.backends.cudnn.version()}")
    else:
        print("  ⚠  Nenhuma GPU detectada — rodando em CPU")


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, epoch, use_amp):
    model.train()
    epoch_loss = 0.0
    steps = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        steps += 1

    avg_loss = epoch_loss / max(steps, 1)
    return avg_loss


def validate(model, loader, loss_fn, dice_metric, post_pred, post_label, device, use_amp):
    model.eval()
    val_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            val_loss += loss.item()
            steps += 1

            # Dice metric (em float32 para precisão)
            preds = post_pred(outputs.float())
            labels_oh = post_label(labels.float())
            dice_metric(y_pred=preds, y=labels_oh)

    avg_loss = val_loss / max(steps, 1)
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()

    return avg_loss, dice_score


def main():
    parser = argparse.ArgumentParser(description="RadianceAI — Treinamento de Segmentação")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    # Carregar config
    cfg = load_config(args.config)
    print("=" * 60)
    print("  RadianceAI — Training Pipeline (GPU Optimized)")
    print("=" * 60)
    print(f"  Config: {args.config}")
    for k, v in cfg.items():
        print(f"    {k}: {v}")

    image_size = tuple(cfg["image_size"])
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["learning_rate"]
    train_manifest = cfg["train_manifest"]
    val_manifest = cfg["val_manifest"]
    checkpoint_name = cfg["checkpoint_name"]
    num_classes = cfg.get("num_classes", 2)
    num_workers = cfg.get("num_workers", 4)
    use_amp = cfg.get("amp", True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    print_gpu_info()

    # Otimizações CUDA
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  ✅ cuDNN benchmark ON | TF32 ON | AMP:", "ON" if use_amp else "OFF")

    # Datasets & Loaders
    print("\n📂 Carregando datasets...")
    train_ds = ManifestDataset(train_manifest, get_train_transforms(image_size))
    val_ds = ManifestDataset(val_manifest, get_val_transforms(image_size))

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )

    print(f"  Train: {len(train_ds)} amostras | batch={batch_size} | workers={num_workers}")
    print(f"  Val:   {len(val_ds)} amostras")

    # Modelo, Loss, Optimizer, Metrics
    model = build_segmentation_model(num_classes=num_classes).to(device)
    loss_fn = get_loss_function(num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=use_amp)
    dice_metric = get_dice_metric(num_classes=num_classes)
    post_pred, post_label = get_post_transforms(num_classes=num_classes)

    # Compilar modelo (PyTorch 2.x)
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("  ✅ torch.compile ativado")
        except Exception:
            print("  ⚠  torch.compile não disponível, usando eager mode")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Modelo: DynUNet ({num_classes} classes, {param_count:,} params)")
    print(f"   Loss: DiceCELoss | Optimizer: AdamW (lr={lr})")
    print(f"   Scheduler: CosineAnnealingLR | AMP: {'ON' if use_amp else 'OFF'}")

    # Checkpoint dir
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / checkpoint_name

    # Training loop
    best_dice = 0.0
    print(f"\n🚀 Iniciando treino por {epochs} épocas...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, epoch, use_amp
        )
        val_loss, dice_score = validate(
            model, val_loader, loss_fn, dice_metric, post_pred, post_label, device, use_amp
        )

        scheduler.step()
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        gpu_mem = ""
        if device.type == "cuda":
            mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            gpu_mem = f" | GPU Mem: {mem_gb:.1f}GB"

        print(
            f"  [Epoch {epoch:03d}/{epochs}] "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Dice: {dice_score:.4f} | LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s{gpu_mem}"
        )

        # Salvar melhor modelo
        if dice_score > best_dice:
            best_dice = dice_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_dice": best_dice,
                "config": cfg,
            }, ckpt_path)
            print(f"  💾 Checkpoint salvo: {ckpt_path.name} (Dice={best_dice:.4f})")

    # Resumo final
    print(f"\n{'=' * 60}")
    print(f"  ✅ Treino finalizado! Melhor Dice: {best_dice:.4f}")
    print(f"  📁 Checkpoint: {ckpt_path}")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"  🎯 Pico de VRAM: {peak:.2f} GB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
