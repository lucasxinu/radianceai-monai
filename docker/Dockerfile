# ──────────────────────────────────────────────
# RadianceAI MONAI — Otimizado para RunPod GPUs
# Base: NVIDIA PyTorch (CUDA 12.4 + cuDNN 9)
# ──────────────────────────────────────────────
FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    CUDA_MODULE_LOADING=LAZY

WORKDIR /app

# Instalar dependências do sistema (opencv precisa de libs gráficas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python (torch já vem na imagem base)
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar projeto
COPY . .

# Preparar dataset ao buildar (opcional, pode rodar manualmente)
# RUN python scripts/prepare_dataset.py

CMD ["python", "-m", "training.run_training"]
