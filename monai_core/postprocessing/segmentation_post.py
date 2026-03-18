from monai.transforms import Compose, Activations, AsDiscrete


def get_post_transforms(num_classes=2):
    """
    Pós-processamento das predições do modelo:
      1. Softmax para probabilidades
      2. Argmax para máscara discreta
    """
    post_pred = Compose([
        Activations(softmax=True),
        AsDiscrete(argmax=True),
    ])

    post_label = Compose([
        AsDiscrete(to_onehot=num_classes),
    ])

    return post_pred, post_label
