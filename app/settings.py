import os
from dataclasses import dataclass

@dataclass
class Settings:
    environment: str = os.getenv("ENVIRONMENT", "development")
    data_root: str = os.getenv("DATA_ROOT", "./datasets")
    checkpoints_root: str = os.getenv("CHECKPOINTS_ROOT", "./checkpoints")
    logs_root: str = os.getenv("LOGS_ROOT", "./logs")
    device: str = os.getenv("DEVICE", "cuda")
    app_name: str = "RadianceAI.MONAI"

settings = Settings()
