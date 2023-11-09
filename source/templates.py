from collections import deque
import numpy as np
from pydantic import BaseModel, Field


class SuspiciousObject(BaseModel):
    """
    Структура данных для найденного, но не подтвержденного, как оставленный,
        временно статического в кадре предмета.
    :param observation_counter: Счетчик кадров наблюдения за объектом.
    :param disappearance_counter: Счетчик кадров отсутствия объекта.
    :param contour_area: Площадь контура объекта, взятая из маски временно статических объектов.
    :param bbox_coordinates: Координаты bbox'а объекта.
    :param centroid_coordinates: Координаты центроида объекта.
    :param updated: Флаг, отвечающий за то, что объект на текущем кадре обновился
        (нужен, чтобы обновлять счетчик отсутствия в кадре).
    :param unattended: Подтверждение, что подозрительный предмет - оставленный.
    """
    observation_counter: int = 900  # примерно 30 секунд уже находится в кадре
    disappearance_counter: int = 120  # примерно 4 секунды
    contour_area: int = 0
    bbox_coordinates: np.array = Field(default_factory=lambda: np.zeros(4))
    centroid_coordinates: np.array = Field(default_factory=lambda: np.zeros(2))
    updated: bool = True
    unattended: bool = False

    class Config:
        """Для того чтобы была возможность объявлять поля как numpy array."""
        arbitrary_types_allowed = True

    def update(self, **new_data):
        """
        Обновление данных.
        :param new_data: Новые данные.
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
                case 'unattended':
                    self.unattended = value


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
