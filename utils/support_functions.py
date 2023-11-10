"""
Вспомогательные функции
"""

import numpy as np
import torch
from torchvision import ops


def iou(bbox1: np.array, bbox2: np.array) -> np.float32:
    """
    Вычисляет IOU двух входных bbox'ов.
    :param bbox1: Координаты 1 bbox'a в формате np.array([x1, y1, x2, y2]).
    :param bbox2: Координаты 2 bbox'a в формате np.array([x1, y1, x2, y2]).
    :return: IOU.
    """
    return ops.box_iou(
        torch.tensor([bbox1], dtype=torch.float),
        torch.tensor([bbox2], dtype=torch.float),
    ).numpy()[0][0]
