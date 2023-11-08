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
        (нужен, чтобы обновлять счетчик отсутствия в кадре)
    """
    observation_counter: int = 1000  # тк предмет уже присутствует какое-то время в кадре в момент обнаружения
    disappearance_counter: int = 150  # примерно 5 секунд
    contour_area: int = 0
    bbox_coordinates: np.array = Field(default_factory=lambda: np.zeros(4))
    centroid_coordinates: np.array = Field(default_factory=lambda: np.zeros(2))
    updated: bool = True

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
