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
    :param last_seen_timestamp: Timestamp последнего обнаружения объекта в маске
        (для того, чтобы отслеживать исчезновение из маски).
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
    last_seen_timestamp: time = Field(default_factory=lambda: time.time())
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
                case 'last_seen_timestamp':
                    self.last_seen_timestamp = value
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
    :param confirmation_frame: Кадр подтверждения того, что предмет - оставленный.
    :param obs_loss_timestamp: Timestamp потери объекта в маске.
    :param leaving_frames: История кадров оставления предмета.
        Как только объект появился - записываем историю его появления.
    :param probably_left_object_people: Ббоксы людей, которые, вероятнее всего, оставили предмет.
    :param contour_mask: Бинаризованная маска кадра, в которой белым залит контур объекта, а черным - фон.
    :param bbox_coordinates: Координаты bbox'а объекта.
    :param linked: Флаг, отвечающий за то, что для данного предмета найден человек, оставивший его.
    """
    __config = Config('config.yml')

    object_id: UUID = Field(default_factory=uuid4)
    detection_timestamp: time = Field(default_factory=lambda: time.time())
    confirmation_frame: np.array = np.array([])
    obs_loss_timestamp: time = float('inf')
    leaving_frames: List[np.array] = []
    probably_left_object_people: List[np.array] = []
    contour_mask: np.array = np.array([])
    bbox_coordinates: np.array = Field(default_factory=lambda: np.zeros(4))
    linked: bool = False

    class Config:
        """Для того чтобы была возможность объявлять поля как numpy array."""
        arbitrary_types_allowed = True

    def set_obs_loss_timestamp(self, loss_timestamp) -> None:
        """
        Сеттер для времени ненаблюдения объекта в кадре.
        :param loss_timestamp: Время ненаблюдения.
        :return: None.
        """
        self.obs_loss_timestamp = loss_timestamp

    def set_prob_left(self, prob_left: list | None) -> None:
        """
        Сеттер параметра предполагаемых людей, оставивших предмет.
        :param prob_left: Список кадров предполагаемых людей.
        :return: None.
        """
        self.probably_left_object_people = prob_left if prob_left is not None else []
        self.linked = True  # и сразу же ставим флаг на то, что предмет уже прошел связывание
