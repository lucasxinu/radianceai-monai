from monai.networks.nets import DynUNet


def build_segmentation_model(num_classes=2, in_channels=1):
    """
    Constrói o modelo DynUNet para segmentação 2D.

    Args:
        num_classes: número de classes de saída (inclui background).
        in_channels: canais de entrada (1=grayscale, 3=RGB).
    """
    return DynUNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=num_classes,
        kernel_size=[3, 3, 3],
        strides=[1, 2, 2],
        upsample_kernel_size=[2, 2],
        filters=[32, 64, 128],
    )
