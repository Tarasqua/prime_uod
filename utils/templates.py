from collections import deque
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
import numpy as np

from source.config_loader import Config


class DetectedObject(BaseModel):
    """
    Структура данных для найденного в маске движения предмета, прошедшего фильтрацию по
        площади. Данный предмет имеет два флага - подозрительный и оставленный предмет,
        которые меняют свое состояние в зависимости от времени нахождения в кадре.
    :param object_id: Уникальный id объекта.
    :param observation_counter: Счетчик кадров наблюдения за объектом.
    :param disappearance_counter: Счетчик кадров отсутствия объекта.
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
    observation_counter: int = __config.get('DETECTED_OBJECT', 'DEFAULT_OBSERVATION_COUNTER')
    disappearance_counter: int = __config.get('DETECTED_OBJECT', 'DEFAULT_DISAPPEARANCE_COUNTER')
    contour_area: int = 0
    contour_mask: np.array = np.array([])
    bbox_coordinates: np.array = Field(default_factory=lambda: np.zeros(4))
    centroid_coordinates: np.array = Field(default_factory=lambda: np.zeros(2))
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
                case 'observation_counter':
                    self.observation_counter += value
                case 'disappearance_counter':
                    self.disappearance_counter -= value
                case 'contour_area':
                    self.contour_area = value
                case 'contour_mask':
                    self.contour_mask = value
                case 'bbox_coordinated':
                    self.bbox_coordinates = value
                case 'centroid_coordinates':
                    self.centroid_coordinates = value
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

    def set_obs_counter(self, value):
        """
        Выставление конкретного значения на счетчик наблюдения.
        :param value: Интовое значение.
        :return: None.
        """
        self.observation_counter = value


class UnattendedObject(BaseModel):
    """
    Структура данных для выявленных оставленных предметов.
    :param object_id: Уникальный id объекта.
    :param contour_mask: Бинаризованная маска кадра, в которой белым залит контур объекта, а черным - фон.
    :param bbox_coordinates: Координаты bbox'а объекта.
    :param detection_frame: Кадр, когда объект был впервые обнаружен.
    :param saved: Флаг, отвечающий за то, сохранен ли данный предмет в базу или нет
        (для того, чтобы продолжать заливать маску, но не сохранять больше 1 раза).
    :param fill_black_timeout: Таймаут заливки сегмента черным в маске со временно стат объектами.
    """
    __config = Config('config.yml')

    object_id: UUID = Field(default_factory=uuid4)
    contour_mask: np.array = np.array([])
    bbox_coordinates: np.array = Field(default_factory=lambda: np.zeros(4))
    detection_frame: np.array = np.array([])
    saved: bool = False
    # по дефолту ставим таймаут равным времени, нужному на появление + на подтверждение "оставленности" предмета
    fill_black_timeout: int = (__config.get('UOD', 'SUSPICIOUS_TO_UNATTENDED_TIMEOUT') +
                               __config.get('DETECTED_OBJECT', 'DEFAULT_OBSERVATION_COUNTER'))

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
