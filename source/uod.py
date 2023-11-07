"""
@tarasqua
"""
import os
import time
from pathlib import Path
import asyncio
from typing import List

import cv2
import numpy as np
import ultralytics.engine.results
from ultralytics import YOLO

from config_loader import Config
from background_subtractor import BackgroundSubtractor
from templates import SuspiciousObject


class UOD:
    """
    Unattended Object Detector
    """

    def __init__(self, frame_shape: np.array, frame_dtype: np.dtype, roi: np.array, remove_people: bool = True):
        """
        Детектор оставленных предметов.
        :param frame_shape: Размеры кадра последовательности (cv2 image.shape).
        :param frame_dtype: Тип кадра последовательности (cv2 image.dtype).
        :param roi: Полигон ROI.
        :param remove_people: Удалять из маски движения людей или нет.
        """
        self.roi = roi
        self.remove_people = remove_people  # для отладки или для большей производительности и меньшей точности
        config_ = Config('config.yml')
        self.bg_subtractor = BackgroundSubtractor(frame_shape, frame_dtype, config_.get('BG_SUBTRACTION'), roi)
        self.yolo_seg = self.__set_yolo_model(config_.get('UOD', 'HUMAN_DETECTION', 'YOLO_MODEL')) \
            if remove_people else None
        self.yolo_conf = config_.get('UOD', 'HUMAN_DETECTION', 'YOLO_CONFIDENCE') \
            if remove_people else None
        self.area_threshold = (np.prod(np.array(frame_shape[:-1])) *  # % от площади кадра
                               (config_.get('UOD', 'AREA_THRESH_FRAME_PERCENT') * 0.01))
        self.suspicious_objects: List[SuspiciousObject] = []

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

    async def __process_area(self, centroid: np.array, stat: np.array) -> np.array:
        """
        Обработка связной области, полученной из модели фона:
            - фильтрация по площади;
            - переводи из xywh в xyxy;
            - объединение координат bbox'а с координатами центроида.
        :param centroid: Координаты центроида связной области.
        :param stat: Координаты bbox'а и площадь связной области.
        :return: Массив с координатами центроида и bbox'а области вида np.array([c_x, c_y, x1, y1, x2, y2]).
        """
        if stat[-1] < self.area_threshold:
            return None
        x, y, w, h = stat[:-1]
        return np.concatenate([centroid, [x, y, x + w, y + h]]).astype(int)

    async def __get_mask_data(self, tso_mask: cv2.typing.MatLike) -> np.array:
        """
        Нахождение и фильтрация по площади центроидов и bbox'ов временно статических объектов из маски.
        :param tso_mask: Маска со временно статическими объектами в ней.
        :return: Массив вида np.array([[centroid_x, centroid_y, x1, y1, x2, y2], [...], ...])
        """
        # находим связные области в маске
        connected_areas = cv2.connectedComponentsWithStats(tso_mask, 4, cv2.CV_32S)
        # обрабатываем полученные области и возвращаем их
        process_area_tasks = [asyncio.create_task(self.__process_area(centroid, stat))
                              for centroid, stat in zip(connected_areas[3][1:], connected_areas[2][1:])]
        centroids_bboxes = await asyncio.gather(*process_area_tasks)
        return np.array([bbox for bbox in centroids_bboxes if bbox is not None])

    async def __match_bbox(self, data):
        """
        Сопоставление одной рамки с уже имеющимися по центроидам.
        TODO: сделать более красиво + сделать идентификацю предмета как оставленного + обработку исчезновения.
        :param data: np.array([centroid_x, centroid_y, x1, y1, x2, y2])
        :return: _
        """
        matched = False
        for susp_obj in self.suspicious_objects:
            x1, y1, x2, y2 = susp_obj.bbox_coordinates
            if x1 <= data[0] <= x2 and y1 <= data[1] <= y2:
                susp_obj.update(
                    observation_counter=1, bbox_coordinated=data[2:], centroid_coordinates=data[:2])
                matched = True
        if not matched:
            self.suspicious_objects.append(SuspiciousObject(
                bbox_coordinates=data[2:],
                centroid_coordinates=data[:2]
            ))

    async def __match_mask_data(self, mask_data):
        """
        Сопоставление всех рамок.
        TODO: доделать (описано выше).
        :param mask_data:
        :return:
        """
        if not self.suspicious_objects:
            [self.suspicious_objects.append(
                SuspiciousObject(bbox_coordinates=data[2:], centroid_coordinates=data[:2]))
             for data in mask_data]
        else:
            match_bbox_tasks = [asyncio.create_task(self.__match_bbox(data)) for data in mask_data]
            [await task for task in match_bbox_tasks]

    async def detect_(self, current_frame: np.array):
        """
        Детекция оставленных предметов в последовательности кадров.
        TODO: временно возвращает текущий кадр со всеми временно статическими объектами
        :param current_frame: Текущее изображение последовательности.
        :return: bbox оставленного предмета.
        """
        # start = time.perf_counter()  # считаем FPS
        # получаем bbox'ы из модели фона
        if self.remove_people:
            detections: ultralytics.engine.results = self.yolo_seg.predict(
                current_frame, classes=[0], verbose=False, conf=self.yolo_conf)[0]
            tso_mask = await self.bg_subtractor.get_tso_mask(current_frame, detections.masks)
        else:
            tso_mask = await self.bg_subtractor.get_tso_mask(current_frame, None)
        mask_data = await self.__get_mask_data(tso_mask)
        if mask_data.size != 0:
            await self.__match_mask_data(mask_data)
        # строим все временно статические объекты (временное решение)
        for bbox in self.suspicious_objects:
            x1, y1, x2, y2 = bbox.bbox_coordinates
            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # print(1 / (time.perf_counter() - start))
        return current_frame
