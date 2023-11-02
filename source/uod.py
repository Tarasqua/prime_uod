"""
@tarasqua
"""
import os
import time
from pathlib import Path
import asyncio

import cv2
import numpy as np
import ultralytics.engine.results
from ultralytics import YOLO

from config_loader import Config
from background_subtractor import BackgroundSubtractor


class UOD:
    """
    Unattended Object Detector
    """

    def __init__(self, frame_shape: np.array):
        config_ = Config('config.yml')
        self.bg_subtractor = BackgroundSubtractor(frame_shape, config_.get('BG_SUBTRACTION'))
        self.yolo_seg = self.__set_yolo_model(config_.get('HUMAN_DETECTION', 'YOLO_MODEL'))
        self.yolo_conf = config_.get('HUMAN_DETECTION', 'YOLO_CONFIDENCE')

    @staticmethod
    def __set_yolo_model(yolo_model) -> YOLO:
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

    async def detect_(self, current_frame: np.array):
        """
        Детекция оставленных предметов в последовательности кадров.
        TODO: временно возвращает маску со временно статическими объектами
        :param current_frame: Текущее изображение последовательности.
        :return: bbox оставленного предмета.
        """
        start = time.perf_counter()  # считаем FPS
        detections: ultralytics.engine.results = self.yolo_seg.predict(
            current_frame, classes=[0], verbose=False, conf=self.yolo_conf)[0]
        fg_mask = await self.bg_subtractor.get_temp_stat_objects(current_frame, detections.masks)

        print(1 / (time.perf_counter() - start))
        return fg_mask
