"""
@tarasqua
"""
import asyncio

import cv2
import numpy as np

from source.config_loader import Config


class BackgroundSubtractor:
    """
    Нахождение маски со временно статическими объектами
    """

    def __init__(self, frame_shape: np.array):
        """
        Нахождение маски со временно статическими объектами, путем нахождения абсолютной разницы
            между исходными кадрами без движения, полученными в результате замены пикселей
            движения пикселями заднего плана из моделей с разным временем накопления.
        :param frame_shape: Размеры кадра последовательности (cv2 image.shape).
        """
        config_ = Config('config.yml')
        # Модели вычитания фонов
        self.gsoc_slow = cv2.bgsegm.createBackgroundSubtractorGSOC(
            hitsThreshold=config_.get('BG_SUBTRACTION', 'GSOC_MODELS', 'HITS_THRESHOLD'),
            replaceRate=config_.get('BG_SUBTRACTION', 'GSOC_MODELS', 'REPLACE_RATE_SLOW')
        )
        self.gsoc_fast = cv2.bgsegm.createBackgroundSubtractorGSOC(
            hitsThreshold=config_.get('BG_SUBTRACTION', 'GSOC_MODELS', 'HITS_THRESHOLD'),
            replaceRate=config_.get('BG_SUBTRACTION', 'GSOC_MODELS', 'REPLACE_RATE_FAST')
        )
        self.gsoc_mid = cv2.bgsegm.createBackgroundSubtractorGSOC(
            hitsThreshold=config_.get('BG_SUBTRACTION', 'GSOC_MODELS', 'HITS_THRESHOLD'),
            replaceRate=config_.get('BG_SUBTRACTION', 'GSOC_MODELS', 'REPLACE_RATE_FAST') / 2
        )
        # Морфологическое закрытие для маски движения
        self.morph_close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(config_.get('BG_SUBTRACTION', 'MORPH_CLOSE', 'KERNEL_SIZE')))
        self.morph_close_iterations = config_.get('BG_SUBTRACTION', 'MORPH_CLOSE', 'ITERATIONS')
        # Дополнительные переменные
        self.frame_shape = frame_shape[:-1][::-1]
        self.resize_shape = (np.array(self.frame_shape) /
                             config_.get('BG_SUBTRACTION', 'REDUCE_FRAME_SHAPE_MULTIPLIER')).astype(int)
        self.tso_mask = None

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

    async def get_tso_mask(self, current_frame: np.array) -> np.ndarray:
        """
        TSO - temporary static objects.
        Нахождение маски со временно статическими объектами, при этом выполняя следующие шаги:
            - используя модели вычитания заднего плана с разной скоростью накопления, копит фон;
            - подменяет пиксели в копиях исходного изображения на пиксели заднего плана в тех
                местах, где обнаружено движение. Тем самым, получая два изображения с разной
                скоростью изменения сцены;
            - берет абсолютную разность полученных изображений, блюрит, бинаризирует, а также
                применяет морфологическое закрытие;
            - затем к исходному изображению применяется еще одна модель со скоростью накопления,
                в два раза меньшей скорости накопления быстрой модели;
            - так, для дополнительного удаления шумов в итоговой маске, в тех местах, где есть
                движение в маске со средним значением накопления, удаляется движение из итоговой.
        :param current_frame: Текущий кадр последовательности.
        :return: Бинаризованная маска со временно статическими объектами.
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
        closing = cv2.morphologyEx(  # а также делаем морфологическое закрытие
            fg_mask, cv2.MORPH_CLOSE, self.morph_close_kernel, iterations=self.morph_close_iterations)
        # используем еще одну модель со средней скоростью накопления
        fg_mask_mid = self.gsoc_mid.apply(resized_frame)
        closing_mid = cv2.morphologyEx(  # морфологическое закрытие к ней
            fg_mask_mid, cv2.MORPH_CLOSE, self.morph_close_kernel, iterations=self.morph_close_iterations)
        # убираем движение в итоговой маске в тех местах, где есть движение на средней
        closing[closing_mid == 255] = 0
        return cv2.resize(closing, self.frame_shape)  # в исходный размер
