"""
@tarasqua
"""
import asyncio

import cv2
import numpy as np

from background_subtractor import BackgroundSubtractor
from config_loader import Config
from ultralytics.engine.results import Masks
from ultralytics.utils import ops


class TSODetector:
    """Обработка кадра, нахождение в нем временно статических объектов."""
    def __init__(self, frame_shape: np.array, frame_dtype: np.dtype, roi: list):
        self.frame_shape = frame_shape
        config_ = Config('config.yml')
        self.area_threshold = (np.prod(np.array(frame_shape[:-1])) *  # % от площади кадра
                               (config_.get('TSO', 'AREA_THRESH_FRAME_PERCENT') * 0.01))
        self.con_comp_connectivity = config_.get('TSO', 'CON_COMPONENTS_CONNECTIVITY')
        self.bg_subtractor = BackgroundSubtractor(frame_shape)
        self.roi_stencil = self.__get_roi_mask(frame_shape[:-1], frame_dtype, roi)
        self.tso_mask = None

    @staticmethod
    def __get_roi_mask(frame_shape: tuple[int, int, int], frame_dtype: np.dtype, roi: list) -> np.array:
        """
        Создает маску, залитую черным вне ROI
        :param frame_shape: Размер изображения (height, width, channels).
        :param frame_dtype: Тип изображения (uint8, etc).
        :param roi: Полигон точек вида np.array([[x, y], [x, y], ...]).
        :return: Маска, залитая белым внутри ROI и черным - во вне.
        """
        stencil = np.zeros(frame_shape).astype(frame_dtype)
        cv2.fillPoly(stencil, roi, (255, 255, 255))
        return stencil

    async def __remove_person(self, det_mask: np.array) -> None:
        """
        По сегментационной маске, вычитает человека из маски со временно статическими предметами
        (на случай, если человек задержался в кадре).
        :param det_mask: Бинарная маска, в которой 1 - сегмент человека, 0 - фон.
        :return: None.
        """
        mask = ops.scale_image(det_mask, self.frame_shape[::-1])[:, :, 0]
        self.tso_mask[mask == 1] = 0

    async def __process_area(self, centroid: np.array, stat: np.array) -> np.array:
        """
        Обработка связной области, полученной из модели фона:
            - фильтрация по площади;
            - переводи из xywh в xyxy;
            - объединение координат bbox'а и площади контура с координатами центроида.
        :param centroid: Координаты центроида связной области.
        :param stat: Координаты bbox'а и площадь связной области.
        :return: Массив с координатами центроида и bbox'а области + площади контура
            вида np.array([c_x, c_y, x1, y1, x2, y2, area]).
        """
        if stat[-1] < self.area_threshold:
            return None
        x, y, w, h, area = stat
        return np.concatenate([centroid, [x, y, x + w, y + h, area]]).astype(int)

    async def __get_mask_data(self) -> np.array:
        """
        Нахождение и фильтрация по площади центроидов и bbox'ов временно статических объектов из маски.
        :return: Массив вида np.array([[centroid_x, centroid_y, x1, y1, x2, y2], [...], ...])
        """
        # находим связные области в маске
        connected_areas = cv2.connectedComponentsWithStats(self.tso_mask, self.con_comp_connectivity, cv2.CV_32S)
        # обрабатываем полученные области и возвращаем их
        process_area_tasks = [asyncio.create_task(self.__process_area(centroid, stat))
                              for centroid, stat in zip(connected_areas[3][1:], connected_areas[2][1:])]
        centroids_bboxes = await asyncio.gather(*process_area_tasks)
        return np.array([bbox for bbox in centroids_bboxes if bbox is not None])  # фильтруем None

    async def process_frame(self, current_frame: np.array, det_masks: Masks or None) -> np.array:
        """
        Обработка изображения, нахождение временно статических объектов в кадре.
        :param current_frame: Текущий кадр последовательности.
        :param det_masks: Маски детекций людей в кадре, полученные с помощью YOLO-детектора.
        :return: Массив вида np.array([[centroid_x, centroid_y, x1, y1, x2, y2], [...], ...])
        """
        # получаем маску со временно статическим объектами
        self.tso_mask = await self.bg_subtractor.get_tso_mask(current_frame)
        # применяем ROI к маске
        self.tso_mask = cv2.bitwise_and(self.tso_mask, self.roi_stencil)
        # если есть люди, вычитаем их из маски
        if det_masks is not None:
            remove_person_tasks = [asyncio.create_task(self.__remove_person(mask))
                                   for mask in det_masks.data.numpy()]
            [await task for task in remove_person_tasks]
        # находим объекты в маске
        mask_data = await self.__get_mask_data()
        return mask_data, self.tso_mask  # ДЛЯ ОТЛАДКИ возвращаем и маску
