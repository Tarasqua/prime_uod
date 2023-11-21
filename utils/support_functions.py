"""
Вспомогательные функции
"""
import asyncio
import datetime
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import ops
from ultralytics import YOLO

from utils.templates import UnattendedObject, DetectedObject


def set_yolo_model(yolo_model) -> YOLO:
    """
    Выполняет проверку путей и наличие модели:
        Если директория отсутствует, создает ее, а также скачивает в нее необходимую модель
    :param yolo_model: n (nano), m (medium)...
    :return: Объект YOLO-pose
    """
    yolo_models_path = os.path.join(Path(__file__).resolve().parents[1], 'resources', 'models', 'yolo_models')
    if not os.path.exists(yolo_models_path):
        Path(yolo_models_path).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(yolo_models_path, f'yolov8{yolo_model}-seg')
    if not os.path.exists(f'{model_path}.onnx'):
        YOLO(model_path).export(format='onnx')
    return YOLO(f'{model_path}.onnx')


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
    # создаем новую папку для сохранения кадров
    detections_path = os.path.join(Path(__file__).resolve().parents[1], 'resources', 'uod_detections')
    directory = os.path.join(detections_path,
                             datetime.datetime.fromtimestamp(obj.detection_timestamp).strftime("%B %d, %H:%M:%S"))
    if not os.path.exists(directory):
        os.mkdir(directory)
    # сохраняем кадры в папку
    x1, y1, x2, y2 = obj.bbox_coordinates
    for frame in obj.leaving_frames:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(directory, f'{len(os.listdir(directory)) + 1}.png'), frame)
    # выставляем флаг, что объект сохранен
    obj.update(saved=True)


async def plot_bboxes(detected_objects: list, frame: np.array) -> np.array:
    """
    Отрисовка bbox'ов подозрительных или оставленных предметов.
    :return: Фрейм с отрисованными ббоксами.
    """

    if not detected_objects:
        return frame

    async def plot(object_data: DetectedObject) -> None:
        """Строим один bbox."""
        x1, y1, x2, y2 = object_data.bbox_coordinates
        # подозрительный объект - желтый, оставленный - красный
        color = (30, 255, 255) if not object_data.unattended else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    plot_tasks = [asyncio.create_task(plot(detected_object))
                  for detected_object in detected_objects
                  if detected_object.suspicious or detected_object.unattended]
    [await task for task in plot_tasks]
    return frame
