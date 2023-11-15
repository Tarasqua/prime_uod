"""
Вспомогательные функции
"""
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import ops

from utils.templates import UnattendedObject, DetectedObject


def iou(bbox1: np.array, bbox2: np.array) -> np.float32:
    """
    Вычисляет IOU двух входных bbox'ов.
    :param bbox1: Координаты 1 bbox'a в формате np.array([x1, y1, x2, y2]).
    :param bbox2: Координаты 2 bbox'a в формате np.array([x1, y1, x2, y2]).
    :return: IOU.
    """
    return ops.box_iou(
        torch.from_numpy(np.array([bbox1])),
        torch.from_numpy(np.array([bbox2])),
    ).numpy()[0][0]


async def save_unattended_object(obj: UnattendedObject) -> None:
    """
    Сохраняет кадр, сделанный во время обнаружения предмета и делаем пометку об этом.
    :param obj: Оставленный объект - объект класса UnattendedObject.
    :return: None.
    """
    x1, y1, x2, y2 = obj.bbox_coordinates
    cv2.rectangle(obj.detection_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    detections_path = os.path.join(Path(__file__).resolve().parents[1], 'resources', 'uod_detections')
    im_path = os.path.join(detections_path, f'{len(os.listdir(detections_path)) + 1}.png')
    cv2.imwrite(im_path, obj.detection_frame)
    obj.update(saved=True)  # выставляем флаг, что объект сохранен
