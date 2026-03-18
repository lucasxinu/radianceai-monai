from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd


def get_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
    ])
