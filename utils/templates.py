from collections import deque
import numpy as np
from pydantic import BaseModel, Field
from source.config_loader import Config


class DetectedObject(BaseModel):
    """
    Структура данных для найденного в маске движения предмета, прошедшего фильтрацию по
        площади. Данный предмет имеет два флага - подозрительный и оставленный предмет,
        которые меняют свое состояние в зависимости от времени нахождения в кадре.
    :param observation_counter: Счетчик кадров наблюдения за объектом.
    :param disappearance_counter: Счетчик кадров отсутствия объекта.
    :param contour_area: Площадь контура объекта, взятая из маски временно статических объектов.
    :param bbox_coordinates: Координаты bbox'а объекта.
    :param centroid_coordinates: Координаты центроида объекта.
    :param updated: Флаг, отвечающий за то, что объект на текущем кадре обновился
        (нужен, чтобы обновлять счетчик отсутствия в кадре).
    :param suspicious: Подтверждение, что обнаруженный предмет - подозрительный.
    :param unattended: Подтверждение, что подозрительный предмет - оставленный.
    """
    __config = Config('config.yml')

    observation_counter: int = __config.get('DETECTED_OBJECT', 'DEFAULT_OBSERVATION_COUNTER')
    disappearance_counter: int = __config.get('DETECTED_OBJECT', 'DEFAULT_DISAPPEARANCE_COUNTER')
    contour_area: int = 0
    bbox_coordinates: np.array = Field(default_factory=lambda: np.zeros(4))
    centroid_coordinates: np.array = Field(default_factory=lambda: np.zeros(2))
    updated: bool = True
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

    def reset_dis_counter(self) -> None:
        """
        Выставляет дефолтное значение из конфига для счетчика отсутствия объекта.
        :return: None.
        """
        self.disappearance_counter: int = self.__config.get(
            'DETECTED_OBJECT', 'DEFAULT_DISAPPEARANCE_COUNTER')


class UnattendedObject(BaseModel):
    """
    Структура данных для выявленных оставленных предметов.
    :param bbox_coordinates: Координаты bbox'а объекта.
    :param detection_frame: Кадр, когда объект был впервые обнаружен.
    """
    bbox_coordinates: np.array = Field(default_factory=lambda: np.zeros(4))
    detection_frame: np.array = np.array([])

    class Config:
        """Для того чтобы была возможность объявлять поля как numpy array."""
        arbitrary_types_allowed = True
