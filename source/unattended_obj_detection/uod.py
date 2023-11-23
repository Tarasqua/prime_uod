"""
@tarasqua
"""
from typing import List
from collections import deque

import cv2
import numpy as np
import ultralytics.engine.results

from source.config_loader import Config
from source.temp_stat_obj_detection.tso_detector import TSODetector
from source.unattended_obj_detection.mask_data_matcher import MaskDataMatcher
from source.unattended_obj_detection.exciting_data_updater import DataUpdater
from utils.templates import DetectedObject, UnattendedObject
from utils.support_functions import plot_bboxes, set_yolo_model


class UOD:
    """
    Unattended Object Detector
    """

    def __init__(self, frame_shape: np.array, frame_dtype: np.dtype, roi: List[np.array],
                 dist_zones_points: List[np.array], stream_fps: int, remove_people: bool = True):
        """
        Детектор оставленных предметов.
        :param frame_shape: Размеры кадра последовательности (cv2 image.shape).
        :param frame_dtype: Тип кадра последовательности (cv2 image.dtype).
        :param roi: Список ROI-полигонов.
        :param remove_people: Удалять из маски движения людей или нет.
        """
        self.remove_people = remove_people  # для отладки или для большей производительности и меньшей точности
        config_ = Config('config.yml')
        self.tso_detector = TSODetector(frame_shape, frame_dtype, roi, dist_zones_points)
        self.mask_data_matcher = MaskDataMatcher(
            config_.get('UOD', 'DETECTED_OBJ_AREA_THRESHOLD') / 100,
            config_.get('UOD', 'IOU_THRESHOLD'),
            config_.get('UOD', 'FRAMES_TO_SAVE_NUMBER')
        )
        self.data_updater = DataUpdater(
            config_.get('UOD', 'DETECTED_TO_SUSPICIOUS_TIMEOUT'),
            (config_.get('UOD', 'DETECTED_TO_SUSPICIOUS_TIMEOUT') +
             config_.get('UOD', 'SUSPICIOUS_TO_UNATTENDED_TIMEOUT')),
            config_.get('UOD', 'DISAPPEARANCE_TIMEOUT')
        )
        self.yolo_seg = set_yolo_model(config_.get('UOD', 'HUMAN_DETECTION', 'YOLO_MODEL')) \
            if remove_people else None
        self.yolo_conf = config_.get('UOD', 'HUMAN_DETECTION', 'YOLO_CONFIDENCE') \
            if remove_people else None

        self.detected_objects: List[DetectedObject] = []  # обнаруженные в маске движения предметы
        self.unattended_objects: List[UnattendedObject] = []  # оставленные предметы
        self.history_frames = deque(
            maxlen=int(config_.get('UOD', 'HISTORY_ACCUMULATION_TIME') * stream_fps))

    async def detect_(self, current_frame: np.array) -> np.array:
        """
        Нахождение оставленных предметов в последовательности кадров.
        :param current_frame: Текущее изображение последовательности.
        :return: ДЛЯ ДЕМОНСТРАЦИИ кадры с отрисованными подозрительными (желтая рамка)
            и оставленными предметами (красная рамка).
        """
        # копим историю кадров
        self.history_frames.append(current_frame.copy())
        # обновялем оставленные и берем маски
        self.unattended_objects, unattended_masks = \
            await self.data_updater.update_unattended_objects(self.detected_objects, self.unattended_objects) if (
                self.unattended_objects) else ([], None)
        # получаем bbox'ы из модели фона и удаляем из маски подтвержденные оставленные предметы, чтобы детекция не
        # сработала повторно на часть объекта, образовавшегося от распада полного на несколько составляющих
        if self.remove_people:  # плюс удаляем из маски людей, если это требуется
            detections: ultralytics.engine.results = self.yolo_seg.predict(
                current_frame, classes=[0], verbose=False, conf=self.yolo_conf)[0]
            mask_data, tso_mask = await self.tso_detector.process_frame(
                current_frame, detections.masks, unattended_masks)
        else:
            mask_data, tso_mask = await self.tso_detector.process_frame(
                current_frame, None, unattended_masks)
        # сопоставляем новые с уже имеющимися
        self.detected_objects = await self.mask_data_matcher.match_mask_data(
            mask_data, self.detected_objects, list(self.history_frames))
        if self.detected_objects:
            # обновляем обнаруженные предметы + проверяем, не стал ли какой-либо предмет оставленным по таймауту
            self.detected_objects, self.unattended_objects = \
                await self.data_updater.update_detected_objects(self.detected_objects, self.unattended_objects)
            # удаляем возможные дубликаты подозрительных предметов
            self.detected_objects = await self.data_updater.check_suspicious_duplicates(self.detected_objects)
        # отрисовываем подозрительные и/или оставленные объекты (временное решение)
        current_frame = await plot_bboxes(self.detected_objects, current_frame)
        # return current_frame
        return np.concatenate([current_frame, cv2.merge((tso_mask, tso_mask, tso_mask))], axis=1)
