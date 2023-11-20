import time
from collections import deque
from uuid import UUID, uuid4
from typing import List

import cv2
from pydantic import BaseModel, Field
import numpy as np

from source.config_loader import Config


class DetectedObject(BaseModel):
    """
    Структура данных для найденного в маске движения предмета, прошедшего фильтрацию по
        площади. Данный предмет имеет два флага - подозрительный и оставленный предмет,
        которые меняют свое состояние в зависимости от времени нахождения в кадре.
    :param object_id: Уникальный id объекта.
    :param detection_timestamp: Timestamp обнаружения объекта в маске.
    :param disappearance_counter: Счетчик кадров отсутствия объекта.
    :param leaving_frames: История кадров оставления предмета.
        Как только объект появился - записываем историю его появления.
    :param contour_mask: Бинаризованная маска кадра, в которой белым залит контур объекта, а черным - фон.
    :param contour_area: Площадь контура объекта, взятая из маски временно статических объектов.
    :param bbox_coordinates: Координаты bbox'а объекта.
    :param centroid_coordinates: Координаты центроида объекта.
    :param updated: Флаг, отвечающий за то, что объект на текущем кадре обновился
        (нужен, чтобы обновлять счетчик отсутствия в кадре).
    :param suspicious: Подтверждение, что обнаруженный предмет - подозрительный.
    :param unattended: Подтверждение, что подозрительный предмет - оставленный.
    """
    __config = Config('config.yml')

    object_id: UUID = Field(default_factory=uuid4)
    detection_timestamp: time = Field(default_factory=lambda: time.time())
    disappearance_counter: int = __config.get('DETECTED_OBJECT', 'DEFAULT_DISAPPEARANCE_COUNTER')
    leaving_frames: List[np.array] = []
    contour_area: int = 0
    contour_mask: np.array = np.array([])
    bbox_coordinates: np.array = Field(default_factory=lambda: np.zeros(4))
    centroid_coordinates: deque = deque(maxlen=1)
    updated: bool = False
    suspicious: bool = False
    unattended: bool = False

    class Config:
        """Для того чтобы была возможность объявлять поля как numpy array."""
        arbitrary_types_allowed = True

    def update(self, **new_data) -> None:
        """
        Обновление данных.
        :param new_data: Новые данные.
        :return: None.
        """
        for field, value in new_data.items():
            match field:
                case 'disappearance_counter':
                    self.disappearance_counter -= value
                case 'contour_area':
                    self.contour_area = value
                case 'contour_mask':
                    # добавляем к текущему контуру новый
                    self.contour_mask = cv2.bitwise_or(self.contour_mask, value)
                case 'bbox_coordinated':
                    self.bbox_coordinates = value
                case 'centroid_coordinates':
                    self.centroid_coordinates.append(value)
                case 'updated':
                    self.updated = value
                case 'suspicious':
                    self.suspicious = value
                case 'unattended':
                    self.unattended = value

    def set_dis_counter(self, value: int) -> None:
        """
        Выставление конкретного значения на счетчик отсутствия.
        :param value: Интовое значение.
        :return: None.
        """
        self.disappearance_counter = value

    def set_det_timestamp(self, value) -> None:
        """
        Выставление конкретного значения на время обнаружения.
        :param value: Интовое значение.
        :return: None.
        """
        self.detection_timestamp = value

    def set_leaving_frames(self, value: list) -> None:
        """
        Выставление конкретного значения на историю кадров оставления.
        :param value: Последовательность кадров.
        :return: None.
        """
        self.leaving_frames = value


class UnattendedObject(BaseModel):
    """
    Структура данных для выявленных оставленных предметов.
    :param object_id: Уникальный id объекта.
    :param detection_timestamp: Timestamp обнаружения объекта в маске.
    :param leaving_frames: История кадров оставления предмета.
        Как только объект появился - записываем историю его появления.
    :param contour_mask: Бинаризованная маска кадра, в которой белым залит контур объекта, а черным - фон.
    :param bbox_coordinates: Координаты bbox'а объекта.
    :param saved: Флаг, отвечающий за то, сохранен ли данный предмет в базу или нет
        (для того, чтобы продолжать заливать маску, но не сохранять больше 1 раза).
    :param fill_black_timeout: Таймаут заливки сегмента черным в маске со временно стат объектами.
    """
    __config = Config('config.yml')

    object_id: UUID = Field(default_factory=uuid4)
    detection_timestamp: time = Field(default_factory=lambda: time.time())
    leaving_frames: List[np.array] = []
    contour_mask: np.array = np.array([])
    bbox_coordinates: np.array = Field(default_factory=lambda: np.zeros(4))
    saved: bool = False
    # по дефолту ставим таймаут равным времени, нужному на появление + на подтверждение "оставленности" предмета
    fill_black_timeout: int = (__config.get('UOD', 'DETECTED_TO_SUSPICIOUS_TIMEOUT') +
                               __config.get('UOD', 'SUSPICIOUS_TO_UNATTENDED_TIMEOUT')) * 2

    class Config:
        """Для того чтобы была возможность объявлять поля как numpy array."""
        arbitrary_types_allowed = True

    def update(self, **new_data) -> None:
        """
        Обновление данных.
        :param new_data: Новые данные.
        :return: None.
        """
        for field, value in new_data.items():
            match field:
                case 'saved':
                    self.saved = value
                case 'fill_black_timeout':
                    self.fill_black_timeout -= value
