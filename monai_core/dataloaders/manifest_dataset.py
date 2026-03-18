import json
from monai.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, manifest_path, transforms=None):
        with open(manifest_path) as f:
            data = json.load(f)
        super().__init__(data=data, transform=transforms)
