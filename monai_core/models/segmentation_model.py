from monai.networks.nets import DynUNet


def build_segmentation_model():
    return DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        kernel_size=[3, 3, 3],
        strides=[1, 2, 2],
        upsample_kernel_size=[2, 2],
        filters=[32, 64, 128],
    )
