"""
@tarasqua
"""
import asyncio
from typing import List

import cv2
import numpy as np

from utils.support_functions import get_roi_mask
from source.temp_stat_obj_detection.dist_zones_handler import DistZonesHandler
from source.temp_stat_obj_detection.background_subtractor import BackgroundSubtractor
from source.config_loader import Config
from ultralytics.engine.results import Masks
from ultralytics.utils import ops


class TSODetector:
    """Детектор временно статических объектов."""

    def __init__(self, frame_shape: np.array, frame_dtype: np.dtype,
                 roi: List[np.array], dist_zones_points: List[np.array] | None):
        """
        Обработка кадра, нахождение в нем временно статических объектов.
        :param frame_shape: Размеры кадра последовательности (cv2 image.shape).
        :param frame_dtype: Тип кадра последовательности (cv2 image.dtype).
        :param roi: Список ROI-полигонов.
        :param dist_zones_points: Список точек зон дальности.
            Если None, разбиения на зоны дальности нет и берется уменьшение кадра для ближнего плана.
        """
        self.frame_shape = frame_shape
        self.dist_zones_points = dist_zones_points
        config_ = Config('config.yml')
        self.area_threshold = (np.prod(np.array(frame_shape[:-1])) *  # % от площади кадра
                               (config_.get('TSO', 'AREA_THRESH_FRAME_PERCENT') * 0.01))
        self.con_comp_connectivity = config_.get('TSO', 'CON_COMPONENTS_CONNECTIVITY')
        reduce_ = config_.get('BG_SUBTRACTION', 'REDUCE_FRAME_SHAPE_MULTIPLIER')
        if dist_zones_points is not None:  # использовать разбиение на зоны дальности или нет
            self.dist_zones_handler = DistZonesHandler(frame_shape, dist_zones_points)
            self.bg_subtractors = [
                BackgroundSubtractor(shape, multiplier) for shape, multiplier
                in zip(self.dist_zones_handler.get_frames_shapes(),
                       [reduce_['LONG_RANGE'], reduce_['SEMI_LONG_RANGE'], reduce_['CLOSE_RANGE']])]
        else:
            self.bg_subtractor = BackgroundSubtractor(frame_shape, reduce_['CLOSE_RANGE'])
        self.roi_stencil = get_roi_mask(frame_shape[:-1], frame_dtype, roi)

    async def __get_mask_data(self, tso_mask: np.array) -> list:
        """
        Нахождение и фильтрация по площади центроидов и bbox'ов временно статических объектов из маски.
        :param tso_mask: Маска со временно статическими объектами.
        :return: List из tuple'ов вида [(np.array(centroid_x, centroid_y, x1, y1, x2, y2, area), mask), (...), ...]
        """

        def label_mask(labels_mask: np.array, label: int) -> np.array:
            """
            Заливаем маски черным в тех местах, где не текущий лейбл,
                а также заливаем белым в тех, где есть текущий лейбл.
            :param labels_mask: Маска со всем связными областями.
            :param label: Номер лейбла.
            :return: Маска с сегментом связной области текущего лейбла.
            """
            labels_mask[labels_mask != label] = 0
            labels_mask[labels_mask == label] = 255
            return labels_mask

        def process_area(centroid: np.array, stat: np.array, mask: np.array) -> tuple:
            """
            Обработка связной области, полученной из модели фона.
            :param centroid: Координаты центроида связной области.
            :param stat: Координаты bbox'а и площадь связной области.
            :param mask: Бинаризованная маска, где связная область - белая, фон - черный.
            :return: Tuple из массива с координатами центроида и bbox'а области + площади контура
                вида np.array([c_x, c_y, x1, y1, x2, y2, area]) + маска с сегментом объекта.
            """
            if stat[-1] < self.area_threshold:  # фильтрация по площади
                return None, None
            x, y, w, h, area = stat  # распаковываем, чтобы перевести в формат x1, y1, x2, y2
            return np.concatenate([centroid, [x, y, x + w, y + h, area]]).astype(int), mask

        # находим связные области в маске
        connected_areas = cv2.connectedComponentsWithStats(tso_mask, self.con_comp_connectivity, cv2.CV_32S)
        # находим маски для каждой найденной связной области
        masks_tasks = [asyncio.to_thread(label_mask, connected_areas[1].copy(), i)
                       for i in range(1, connected_areas[0])]  # учитываем тот факт, что 0 - это весь кадр
        masks = await asyncio.gather(*masks_tasks)
        # обрабатываем полученные области и возвращаем их
        process_area_tasks = [asyncio.to_thread(process_area, centroid, stat, mask)
                              for centroid, stat, mask in zip(connected_areas[3][1:], connected_areas[2][1:], masks)]
        data_masks = await asyncio.gather(*process_area_tasks)
        return [(data, mask) for data, mask in data_masks if data is not None]  # фильтруем None

    async def __get_dist_zones_tso_mask(self, current_frame: np.array) -> np.array:
        """
        Получение маски со временно статическими объектами:
            - разбиваем текущий кадр на зоны дальности;
            - обрабатываем каждую зону, получая маски со временно статическими объектами;
            - объединяем маски в одну маску с общим крупным планом в разрешении текущего изображения.
        :param current_frame: Текущий кадр последовательности.
        :return: Маска со временно статическими объектами.
        """

        async def get_mask(frame: np.array, model: BackgroundSubtractor):
            """Вспомогательная функция, которая получает маску со временно
                статическим объектами для данной модели."""
            tso_mask = await model.get_tso_mask(frame)
            return tso_mask

        dist_zones_frames = await self.dist_zones_handler.get_dist_zones_frames(current_frame)
        tso_masks_coros = [get_mask(frame, model) for frame, model in zip(dist_zones_frames, self.bg_subtractors)]
        tso_masks = await asyncio.gather(*tso_masks_coros)
        return await self.dist_zones_handler.get_merged_mask(tso_masks)

    async def process_frame(self, current_frame: np.array, det_masks: Masks or None, uo_masks: list or None) -> list:
        """
        Обработка изображения, нахождение временно статических объектов в кадре.
        :param current_frame: Текущий кадр последовательности.
        :param det_masks: Маски детекций людей в кадре, полученные с помощью YOLO-детектора.
        :param uo_masks: Маски подтвержденных оставленных предметов.
        :return: List из tuple'ов вида [(np.array(centroid_x, centroid_y, x1, y1, x2, y2), mask), (...), ...]
        """
        # получаем маску со временно статическим объектами
        tso_mask = await self.__get_dist_zones_tso_mask(current_frame) if self.dist_zones_points is not None \
            else await self.bg_subtractor.get_tso_mask(current_frame)
        # применяем ROI к маске
        tso_mask = cv2.bitwise_and(tso_mask, self.roi_stencil)

        def remove_person(det_mask: np.array) -> None:
            """
            По сегментационной маске, вычитает человека из маски со временно статическими предметами
                (на случай, если человек задержался в кадре).
            :param det_mask: Бинарная маска, в которой 1 - сегмент человека, 0 - фон.
            :return: None.
            """
            mask = ops.scale_image(det_mask, self.frame_shape[::-1])[:, :, 0]
            tso_mask[mask == 1] = 0

        def remove_unattended(uo_mask: np.array) -> None:
            """
            По маске с оставленным предметом, вычитает данный предмет из маски со временно статическими объектами.
            :param uo_mask: Маска с оставленным предметом.
            :return: None.
            """
            tso_mask[uo_mask == 255] = 0

        # если есть люди, вычитаем их из маски
        if det_masks is not None:
            remove_people_tasks = [asyncio.to_thread(remove_person, mask) for mask in det_masks.data.numpy()]
            await asyncio.gather(*remove_people_tasks)
        # если есть оставленные, также вычитаем их из маски
        if uo_masks is not None:
            remove_unattended_tasks = [asyncio.to_thread(remove_unattended, uo_mask) for uo_mask in uo_masks]
            await asyncio.gather(*remove_unattended_tasks)
        # находим объекты в маске
        mask_data = await self.__get_mask_data(tso_mask)
        return mask_data
