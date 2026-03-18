#!/bin/bash
set -e

echo "============================================"
echo "  RadianceAI MONAI — RunPod Runner"
echo "============================================"

# Detectar diretório do projeto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo ""
echo "📁 Project root: $PROJECT_ROOT"

# GPU Info
echo ""
echo "🔍 GPU Info:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')"
else
    echo "  ⚠ Nenhuma GPU detectada"
fi

# Instalar dependências
echo ""
echo "📦 Instalando dependências..."
pip install --quiet --no-cache-dir monai opencv-python-headless numpy pyyaml pillow 2>&1 | tail -1 || true

# Preparar dataset se manifests não existirem
if [ ! -f "$PROJECT_ROOT/datasets/processed/manifests/train_manifest.json" ]; then
    echo ""
    echo "📂 Preparando dataset..."
    python "$PROJECT_ROOT/scripts/prepare_dataset.py"
else
    echo ""
    echo "✅ Dataset já preparado"
fi

# Rodar treinamento
echo ""
echo "🚀 Iniciando treinamento..."
cd "$PROJECT_ROOT"
python -m training.run_training

echo ""
echo "✅ Treinamento finalizado!"
echo "📂 Checkpoints em: $PROJECT_ROOT/checkpoints/"