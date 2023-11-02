"""
@tarasqua
"""
import asyncio

import cv2
import numpy as np
import ultralytics.engine.results
from ultralytics.utils import ops


class BackgroundSubtractor:
    """
    Нахождение временно статических объектов по модели фона
    """

    def __init__(self, frame_shape: np.array, config_data: dict):
        self.gsoc_slow = cv2.bgsegm.createBackgroundSubtractorGSOC(
            hitsThreshold=config_data['GSOC_MODELS']['HITS_THRESHOLD'],
            replaceRate=config_data['GSOC_MODELS']['REPLACE_RATE_SLOW']
        )
        self.gsoc_fast = cv2.bgsegm.createBackgroundSubtractorGSOC(
            hitsThreshold=config_data['GSOC_MODELS']['HITS_THRESHOLD'],
            replaceRate=config_data['GSOC_MODELS']['REPLACE_RATE_FAST']
        )

        self.morph_close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(config_data['MORPH_CLOSE']['KERNEL_SIZE']))
        self.morph_close_iterations = config_data['MORPH_CLOSE']['ITERATIONS']

        self.frame_shape = frame_shape[:-1][::-1]
        self.resize_shape = (np.array(self.frame_shape) / config_data['REDUCE_FRAME_SHAPE_MULTIPLIER']).astype(int)
        self.area_threshold = np.prod(np.array(self.frame_shape)) * 0.005

        self.fg_mask = None

    @staticmethod
    async def __get_frame_wo_movements(
            frame: np.array, model: cv2.bgsegm.BackgroundSubtractorGSOC) -> np.ndarray:
        """
        Подменяет пиксели в переданном изображении на пиксели фона (полученного путем
        применения переданной модели вычитания фона) в тех местах, где есть движение
        в маске переднего плана.
        :param frame: Копия текущего кадра.
        :param model: Модель вычитания фона GSOC OpenCV.
        :return: Изображение без движения.
        """
        fg_mask = model.apply(frame)  # применяем маску
        r, c = np.where(fg_mask == 255)  # пиксели, где есть движение в маске переднего плана
        frame[(r, c)] = model.getBackgroundImage()[(r, c)]  # подмена пикселей в изображении
        return frame

    async def __get_fg_mask(self, current_frame: np.array) -> np.ndarray:
        """
        Находит временно статические объекты, выполняя следующие шаги:
            - используя модели вычитания заднего плана с разной скоростью накопления, копит фон;
            - подменяет пиксели в копиях исходного изображения на пиксели заднего плана в тех
                местах, где обнаружено движение. Тем самым, получая два изображения с разной
                скоростью изменения сцены;
            - берет абсолютную разность полученных изображений, блюрит, бинаризирует, а также
                применяет морфологическое закрытие;
            - в итоге получает маску со временно статическими объектами.
        :param current_frame: Текущее переданное изображение.
        :return: Маска со временно статическими объектами.
        """
        resized_frame = cv2.resize(current_frame, self.resize_shape)  # для большей производительности уменьшаем
        # применяем к копиям кадров быструю и медленную модели фона
        bg_sub_tasks = [asyncio.create_task(self.__get_frame_wo_movements(resized_frame.copy(), model))
                        for model in [self.gsoc_fast, self.gsoc_slow]]
        frames_wo_movements = await asyncio.gather(*bg_sub_tasks)
        diff = cv2.absdiff(*frames_wo_movements)  # берем их разницу
        blured = cv2.medianBlur(diff, 7)  # блюрим
        _, fg_mask = cv2.threshold(  # и бинаризируем с помощью OTSU
            cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY),
            150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        closing = cv2.morphologyEx(  # а также в конце делаем морфологическое закрытие
            fg_mask, cv2.MORPH_CLOSE, self.morph_close_kernel, iterations=self.morph_close_iterations)
        return cv2.resize(closing, self.frame_shape)  # в исходный размер

    async def __remove_person(self, det_mask: np.array):
        """
        По сегментационной маске, вычитает человека из маски со временно статическими предметами
        (на случай, если человек задержался в кадре).
        :param det_mask: Бинарная маска, в которой 1 - сегмент человека, 0 - фон.
        :return: Маска со временно статическими предметами без сегмента человека.
        """
        mask = ops.scale_image(det_mask, self.frame_shape[::-1])[:, :, 0]
        self.fg_mask[mask == 1] = 0

    async def get_temp_stat_objects(
            self, current_frame: np.array, det_masks: ultralytics.engine.results.Masks) -> np.ndarray:
        """
        Возвращает bbox'ы временно статических объектов
        TODO: временно возвращает маску со временно статическими объектами
        :param current_frame:
        :param det_masks:
        :return:
        """
        # получаем маску со временно статическими объектами
        self.fg_mask = await self.__get_fg_mask(current_frame)
        # выпиливаем из нее людей
        if det_masks is not None:
            remove_person_tasks = [asyncio.create_task(self.__remove_person(mask))
                                   for mask in det_masks.data.numpy()]
            [await task for task in remove_person_tasks]

        return self.fg_mask
