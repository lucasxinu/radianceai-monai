from monai.metrics import DiceMetric as MonaiDiceMetric


def get_dice_metric(num_classes=2):
    """
    Retorna a métrica Dice para avaliação de segmentação.
    include_background=False ignora o fundo na média.
    """
    return MonaiDiceMetric(
        include_background=False,
        reduction="mean",
    )
