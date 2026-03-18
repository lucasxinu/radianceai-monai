#!/bin/bash
# ──────────────────────────────────────────────
# RadianceAI MONAI — Script de execução (RunPod)
# ──────────────────────────────────────────────
set -e

echo "============================================"
echo "  RadianceAI MONAI — RunPod Runner"
echo "============================================"

# Verificar GPU
echo ""
echo "🔍 GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  ⚠  nvidia-smi não disponível"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')" 2>/dev/null

# Preparar dataset (se ainda não foi feito)
if [ ! -f "./datasets/processed/manifests/train_manifest.json" ]; then
    echo ""
    echo "📂 Preparando dataset..."
    python scripts/prepare_dataset.py
fi

# Executar treino
echo ""
echo "🚀 Iniciando treinamento..."
python -m training.run_training "$@"
