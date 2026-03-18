from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    RandFlipd,
    RandRotate90d,
    ToTensord,
    Lambdad,
)


def _to_single_channel(img):
    """Converte RGB (3,H,W) para grayscale (1,H,W) via média dos canais."""
    if img.shape[0] == 3:
        return img.mean(dim=0, keepdim=True)
    return img


def get_train_transforms(image_size=(512, 512)):
    """Transforms de treino com augmentation."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys=["image"], func=_to_single_channel),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=image_size),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.5),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms(image_size=(512, 512)):
    """Transforms de validação (sem augmentation)."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys=["image"], func=_to_single_channel),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=image_size),
        ToTensord(keys=["image", "label"]),
    ])
