"""
Вспомогательные функции
"""
import asyncio
import datetime
import os
from pathlib import Path
import functools
import time
from typing import Callable, Any

import cv2
import numpy as np
import torch
from torchvision import ops
from ultralytics import YOLO
from shapely.geometry import Polygon

from utils.templates import UnattendedObject, DetectedObject


def get_roi_mask(frame_shape: tuple[int, int, int], frame_dtype: np.dtype, roi: list) -> np.array:
    """
    Создает маску, залитую черным вне ROI.
    :param frame_shape: Размер изображения (height, width, channels).
    :param frame_dtype: Тип изображения (uint8, etc).
    :param roi: Полигон точек вида np.array([[x, y], [x, y], ...]).
    :return: Маска, залитая белым внутри ROI и черным - во вне.
    """
    stencil = np.zeros(frame_shape).astype(frame_dtype)
    [cv2.bitwise_or(stencil, s)  # применяем каждый полигон к маске
     for s in [cv2.fillPoly(stencil, [r], (255, 255, 255))
               for r in roi]]
    return stencil


def set_yolo_model(yolo_model: str, yolo_class: str, task: str = 'detect') -> YOLO:
    """
    Выполняет проверку путей и наличие модели:
        Если директория отсутствует, создает ее, а также скачивает в нее необходимую модель
    :param yolo_model: n (nano), m (medium), etc.
    :param yolo_class: seg, pose, boxes
    :param task: detect, segment, classify, pose
    :return: Объект YOLO-pose
    """
    yolo_class = f'-{yolo_class}' if yolo_class != 'boxes' else ''
    yolo_models_path = os.path.join(Path(__file__).resolve().parents[1], 'resources', 'models', 'yolo_models')
    if not os.path.exists(yolo_models_path):
        Path(yolo_models_path).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(yolo_models_path, f'yolov8{yolo_model}{yolo_class}')
    if not os.path.exists(f'{model_path}.onnx'):
        YOLO(model_path).export(format='onnx')
    return YOLO(f'{model_path}.onnx', task=task)


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
    x1, y1, x2, y2 = obj.bbox_coordinates
    # сохраняем предполагаемых оставителей
    for i, human_frame in enumerate(obj.probably_left_object_people):
        cv2.imwrite(os.path.join(directory, f'suspicious_{i}.png'), human_frame)
    # момент оставления
    cv2.imwrite(os.path.join(directory, f'leaving_moment.png'), obj.leaving_frames[0])
    # момент подтверждения
    cv2.imwrite(os.path.join(directory, f'confirmation_moment.png'),
                cv2.rectangle(obj.confirmation_frame.copy(), (x1, y1), (x2, y2), (0, 0, 255), 2))


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


def cart2pol(x, y) -> tuple[float, float]:
    """
    Перевод декартовых координат в полярные.
    :param x: Координата x.
    :param y: Координата y.
    :return: rho (радиус), phi (угол).
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi) -> tuple[float, float]:
    """
    Перевод полярных координат в декартовы.
    :param rho: Координата x
    :param phi: Координата y
    :return: Координаты по x и y
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def inflate_polygon(polygon_points: np.array, scale_multiplier: float) -> np.array:
    """
    Раздувает полигон точек.
    :param polygon_points: Полигон точек вида np.array([[x, y], [x, y], ...]).
    :param scale_multiplier: Во сколько раз раздуть рамку.
    :return: Раздутый полигон того же вида, что и входной.
    """
    centroid = Polygon(polygon_points).centroid
    inflated_polygon = []
    for point in polygon_points:
        rho, phi = cart2pol(point[0] - centroid.x, point[1] - centroid.y)
        x, y = pol2cart(rho * scale_multiplier, phi)
        inflated_polygon.append([x + centroid.x, y + centroid.y])
    return np.array(inflated_polygon)


def async_timed():
    """Декоратор для подсчета времени выполнения асинхронной функции"""
    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapped(*args, **kwargs) -> Any:
            print(f'starting {func} with args {args} {kwargs}')
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                end = time.time()
                total = end - start
                print(f'finished {func} in {total:.4f} second(s)')

        return wrapped

    return wrapper
