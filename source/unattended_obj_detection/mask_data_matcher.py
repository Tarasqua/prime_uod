import asyncio
import time
from operator import itemgetter

import numpy as np

from utils.templates import DetectedObject
from utils.support_functions import iou


class MaskDataMatcher:
    """Вспомогательный класс для сопоставления контуров из маски с теми, что уже есть в базе."""

    def __init__(self, det_obj_area_threshold: float, iou_threshold: float, frames_to_save: int):
        """
        :param det_obj_area_threshold: Порог по площади объекта в маске.
        :param iou_threshold: Порог по IOU для сопоставления объектов в маске.
        :param frames_to_save: Количество кадров, которые будут сохранены в базе.
        """
        self.det_obj_area_threshold = det_obj_area_threshold
        self.iou_threshold = iou_threshold
        self.detected_objects = []
        self.history_frames = []
        self.frames_to_save = frames_to_save

    async def __match_new_object(self, new_object_data: np.array, new_object_mask: np.array) -> None:
        """
        Сопоставление нового полученного объекта с уже имеющимися в базе обнаруженных по IOU.
        :param new_object_data: Данные по объекту из маски в формате
            np.array([centroid_x, centroid_y, x1, y1, x2, y2, area])
        :param new_object_mask:
        :return: None.
        """

        def update_exciting(exciting_: DetectedObject, new_: np.array) -> None:
            """
            Обновление уже существующего в базе объекта.
            :param exciting_: Существующий в базе объект.
            :param new_: Объект, который был сопоставлен.
            :return: None.
            """
            # в любом случае обновляем последнее время наблюдения и то, что объект был сопоставлен
            exciting_.update(last_seen_timestamp=time.time(), updated=True)
            # перестаем обновлять другие параметры как только объект стал оставленным
            if not exciting_.unattended:
                if exciting_.suspicious:  # как только стал подозрительным, т.е. +- устаканился,
                    # и при условии, что площадь не сильно скакнула
                    # (площадь сегмента скачет, если рядом возникло шумовое движение)...
                    if (abs(new_[0][-1] - exciting_.contour_area) / exciting_.contour_area
                            < self.det_obj_area_threshold):
                        # ...обновляем bbox, центроид, площадь + копим маску
                        exciting_.update(contour_area=new_[0][-1], contour_mask=new_[1],
                                         bbox_coordinated=new_[0][2:-1], centroid_coordinates=new_[0][:2])
                else:  # если же предмет пока только формируется, обновляем все параметры
                    exciting_.update(contour_area=new_[0][-1], bbox_coordinated=new_[0][2:-1],
                                     centroid_coordinates=new_[0][:2])

        # находим iou между текущим новым объектом и теми, что уже были обнаружены
        new_detected_iou = [(idx, iou(new_object_data[2:-1], detected_object.bbox_coordinates))
                            for idx, detected_object in enumerate(self.detected_objects)]
        # из них находим те, что проходят по порогу
        thresh_filtered_iou = [(idx, v) for idx, v in new_detected_iou if v > self.iou_threshold]
        if thresh_filtered_iou:  # если какой-либо объект прошел по порогу
            max_iou_idx = max(thresh_filtered_iou, key=itemgetter(1))[0]  # индекс объекта, с которым max IOU
            update_exciting(self.detected_objects[max_iou_idx],
                            (new_object_data, new_object_mask))  # обновляем его
        else:  # если же ни один по порогу не прошел => это новый объект => добавляем в базу
            self.detected_objects.append(
                DetectedObject(
                    contour_area=new_object_data[-1], bbox_coordinates=new_object_data[2:-1],
                    contour_mask=new_object_mask,
                    leaving_frames=self.history_frames[::int(len(self.history_frames) / self.frames_to_save)])
            )

    async def match_mask_data(self, mask_data: np.array, detected_objects: list, history_frames: list) -> (
            list)[DetectedObject]:
        """
        Сопоставление новых данных из маски с уже имеющимися в базе обнаруженных предметов.
        :param mask_data: Все данные, полученные из маски в формате
            np.array([[centroid_x, centroid_y, x1, y1, x2, y2, area], [...]]).
        :param detected_objects: Список объектов, которые нужно обновить.
        :param history_frames: Список кадров истории.
        :return: Обновленный список обнаруженных предметов.
        """
        # если список пустой, добавляем все объекты
        if not detected_objects:
            [detected_objects.append(
                DetectedObject(
                    contour_area=data[-1], bbox_coordinates=data[2:-1], contour_mask=mask,
                    leaving_frames=history_frames[::int(len(history_frames) / self.frames_to_save)]))
                for data, mask in mask_data]
            return detected_objects

        # если не пустой, и если временно статических объектов в кадре не найдено
        if not mask_data:
            # выставляем флаг обновления на текущем кадре у всех объектов в базе на False
            for detected_object in detected_objects:
                detected_object.update(updated=False)
            return detected_objects

        # если же в кадре есть временно статические объекты
        self.detected_objects = detected_objects
        self.history_frames = history_frames
        # связываем только что полученные с уже имеющимися объектами
        match_bbox_tasks = [asyncio.create_task(self.__match_new_object(data, mask))
                            for data, mask in mask_data]
        [await task for task in match_bbox_tasks]
        return self.detected_objects
