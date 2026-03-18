import torch
from torch.utils.data import DataLoader
from monai_core.models.segmentation_model import build_segmentation_model
from monai_core.dataloaders.manifest_dataset import ManifestDataset
from monai_core.transforms.train_transforms import get_transforms


def main():
    ds = ManifestDataset("./datasets/samples/sample_manifest.json", get_transforms())
    loader = DataLoader(ds, batch_size=1)


if __name__ == "__main__":
    main()
