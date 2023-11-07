from collections import deque
import numpy as np
from pydantic import BaseModel, Field


class SuspiciousObject(BaseModel):
    """
    Структура данных для найденного, но не подтвержденного, как оставленный,
    временно статического в кадре предмета.
    """
    observation_counter: int = 1000
    disappearance_counter: int = 120
    bbox_coordinates: np.array = Field(default_factory=lambda: np.zeros(4))
    centroid_coordinates: deque = deque(maxlen=25)

    class Config:
        arbitrary_types_allowed = True

    def update(self, **new_data):
        """
        Обновление данных.
        :param new_data: Новые данные.
        """
        for field, value in new_data.items():
            match field:
                case 'observation_counter':
                    self.observation_counter += 1
                case 'disappearance_counter':
                    self.disappearance_counter -= 1
                case 'bbox_coordinated':
                    self.bbox_coordinates = value
                case 'centroid_coordinates':
                    self.centroid_coordinates.append(value)
