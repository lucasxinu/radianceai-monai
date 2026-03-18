from monai.losses import DiceCELoss


def get_loss_function(num_classes=2):
    """
    Retorna a loss combinada Dice + CrossEntropy.
    Ideal para segmentação com classes desbalanceadas.
    """
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )
