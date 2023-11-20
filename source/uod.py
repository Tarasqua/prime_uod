"""
@tarasqua
"""
import os
import time
from pathlib import Path
import asyncio
from typing import List
from collections import deque
from itertools import combinations
from operator import itemgetter

import cv2
import numpy as np
import ultralytics.engine.results
from ultralytics import YOLO

from config_loader import Config
from tso_detector import TSODetector
from utils.templates import DetectedObject, UnattendedObject
from utils.support_functions import iou, save_unattended_object


class UOD:
    """
    Unattended Object Detector
    """

    def __init__(self, frame_shape: np.array, frame_dtype: np.dtype, roi: list,
                 stream_fps: int, remove_people: bool = True):
        """
        Детектор оставленных предметов.
        :param frame_shape: Размеры кадра последовательности (cv2 image.shape).
        :param frame_dtype: Тип кадра последовательности (cv2 image.dtype).
        :param roi: Список ROI-полигонов.
        :param remove_people: Удалять из маски движения людей или нет.
        """
        self.roi = roi
        self.remove_people = remove_people  # для отладки или для большей производительности и меньшей точности
        config_ = Config('config.yml')
        self.tso_detector = TSODetector(frame_shape, frame_dtype, roi)
        self.yolo_seg = self.__set_yolo_model(config_.get('UOD', 'HUMAN_DETECTION', 'YOLO_MODEL')) \
            if remove_people else None
        self.yolo_conf = config_.get('UOD', 'HUMAN_DETECTION', 'YOLO_CONFIDENCE') \
            if remove_people else None

        self.detected_objects: List[DetectedObject] = []  # обнаруженные в маске движения предметы
        self.iou_threshold = config_.get('UOD', 'IOU_THRESHOLD')
        # пороговое значение (в %) изменения площади для изменения координат рамки
        self.det_obj_area_threshold = config_.get('UOD', 'DETECTED_OBJ_AREA_THRESHOLD') / 100
        self.unattended_objects: List[UnattendedObject] = []  # оставленные предметы
        # сколько кадров должен пролежать предмет для подтверждения того, что он - подозрительный, с учетом
        # времени, потраченного на его обнаружение
        self.suspicious_timeout = config_.get('UOD', 'DETECTED_TO_SUSPICIOUS_TIMEOUT')
        # сколько кадров должен пролежать предмет для подтверждения того, что он - оставленный
        self.unattended_timeout = (config_.get('UOD', 'DETECTED_TO_SUSPICIOUS_TIMEOUT') +
                                   config_.get('UOD', 'SUSPICIOUS_TO_UNATTENDED_TIMEOUT'))
        self.history_frames = deque(maxlen=int(config_.get('UOD', 'HISTORY_ACCUMULATION_TIME') * stream_fps))
        self.frames_to_save = config_.get('UOD', 'FRAMES_TO_SAVE_NUMBER')
        self.frame = None

    @staticmethod
    def __set_yolo_model(yolo_model) -> YOLO:
        """
        Выполняет проверку путей и наличие модели:
            Если директория отсутствует, создает ее, а также скачивает в нее необходимую модель
        :param yolo_model: n (nano), m (medium)...
        :return: Объект YOLO-pose
        """
        yolo_models_path = os.path.join(Path(__file__).resolve().parents[1], 'resources', 'models', 'yolo_models')
        if not os.path.exists(yolo_models_path):
            Path(yolo_models_path).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(yolo_models_path, f'yolov8{yolo_model}-seg')
        if not os.path.exists(f'{model_path}.onnx'):
            YOLO(model_path).export(format='onnx')
        return YOLO(f'{model_path}.onnx')

    async def __match_new_object(self, new_object_data: np.array, new_object_mask: np.array) -> None:
        """
        Сопоставление нового полученного объекта с уже имеющимися в базе обнаруженных по IOU.
        :param new_object_data: Данные по объекту из маски в формате
            np.array([centroid_x, centroid_y, x1, y1, x2, y2, area])
        :param new_object_mask:
        :return: None.
        """

        def update_exciting(exciting_: DetectedObject, new_: np.array) -> None:
            """Обновление уже существующего в базе объекта."""
            # если площадь изменилась на n% и n - больше порогового, а также данный предмет еще не оставленный
            if abs(new_[0][-1] - exciting_.contour_area) / exciting_.contour_area > self.det_obj_area_threshold \
                    and not exciting_.unattended:
                exciting_.update(contour_area=new_[0][-1], bbox_coordinated=new_[0][2:-1],
                                 centroid_coordinates=new_[0][:2], updated=True)
                # а также начинаем копить маску только после того, как объект станет подозрительным
                if exciting_.suspicious:
                    exciting_.update(contour_mask=new_[1])
            else:
                exciting_.update(centroid_coordinates=new_[0][:2], updated=True)

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
                    leaving_frames=list(self.history_frames)[::int(len(self.history_frames) / self.frames_to_save)])
            )

    async def __match_mask_data(self, mask_data: np.array) -> None:
        """
        Сопоставление только что полученных данных из маски, а также обновление уже существующих в базе данных.
        :param mask_data: Все данные, полученные из маски в формате
            np.array([[centroid_x, centroid_y, x1, y1, x2, y2, area], [...]]).
        :return: None.
        """
        if not self.detected_objects:  # если список пустой, добавляем все объекты
            [self.detected_objects.append(
                DetectedObject(
                    contour_area=data[-1], bbox_coordinates=data[2:-1], contour_mask=mask,
                    leaving_frames=list(self.history_frames)[::int(len(self.history_frames) / self.frames_to_save)]))
                for data, mask in mask_data]
        else:  # если не пустой
            # и если временно статических объектов в кадре не найдено
            if not mask_data:
                # выставляем флаг обновления на текущем кадре у всех объектов в базе на False
                for detected_object in self.detected_objects:
                    detected_object.update(updated=False)
            else:  # если же в кадре есть временно статические объекты
                # связываем только что полученные с уже имеющимися объектами
                match_bbox_tasks = [asyncio.create_task(self.__match_new_object(data, mask))
                                    for data, mask in mask_data]
                [await task for task in match_bbox_tasks]

    async def __update_detected_objects(self) -> None:
        """
        Проверяем по таймауту время наблюдения за обнаруженными объектами и, в случае
            прохождения проверки, в зависимости от времени наблюдения, помечаем объект как
            подозрительный или оставленный и, в случае выявления последнего, добавляем к оставленным.
        Также обновляем счетчик отсутствия и удаляем те объекты, в которых счетчик достиг нуля
            (то есть предмета больше в кадре нет).
        :return: None.
        """

        async def update_object(detected_object: DetectedObject) -> None:
            """Обновление одного объекта."""
            # проверка по таймауту на подозрительно долгое пребывание в кадре
            obs_time = time.time() - detected_object.detection_timestamp
            if obs_time >= self.suspicious_timeout and \
                    not detected_object.suspicious:
                # помечаем его как подозрительный
                detected_object.update(suspicious=True)
            elif obs_time >= self.unattended_timeout and \
                    not detected_object.unattended:
                # добавляем в список с оставленными с наследованием id
                self.unattended_objects.append(
                    UnattendedObject(
                        object_id=detected_object.object_id, contour_mask=detected_object.contour_mask,
                        bbox_coordinates=detected_object.bbox_coordinates,
                        leaving_frames=detected_object.leaving_frames,
                        detection_timestamp=detected_object.detection_timestamp
                    ))
                # помечаем его как оставленный в списке обнаруженных
                detected_object.update(unattended=True)
            # обновление счетчика отсутствия (убавляем, если объект не был сопоставлен в текущем кадре)
            if not detected_object.updated:
                detected_object.update(disappearance_counter=1)
            else:
                # если же объект был сопоставлен в текущем кадре, меняем флаг на False для следующего кадра
                detected_object.update(updated=False)

        # проверяем и обновляем обнаруженные
        check_detected_tasks = [asyncio.create_task(update_object(detected_object))
                                for detected_object in self.detected_objects]
        [await task for task in check_detected_tasks]
        # удаляем унесенные объекты
        self.detected_objects = [detected_object for detected_object in self.detected_objects
                                 if detected_object.disappearance_counter != 0]

    async def __check_suspicious_duplicates(self) -> None:
        """
        Проверка и удаление дубликатов подозрительных предметов:
            Так как крупные предметы проявляются в маске постепенно => могут дублироваться; а так как
            проверять все обнаруженные объекты не имеет смысла, ввиду того, что они появляются и пропадают
            достаточно быстро и большинство из них не сопоставится, проверяем только подозрительные.
        :return: None.
        """

        async def check_pair(obj1: DetectedObject, obj2: DetectedObject) -> DetectedObject:
            """Смотрим, пересекаются ли объекты и возвращаем меньший по площади"""
            if iou(obj1.bbox_coordinates, obj2.bbox_coordinates) > 0:
                # находим раннее время обнаружения
                min_det_timestamp = min(obj1.detection_timestamp, obj2.detection_timestamp)
                # берем меньший по площади - его и отфильтруем далее
                if obj1.contour_area < obj2.contour_area:
                    # большему присваиваем раннее время наблюдения, больший счетчик отсутствия, так как
                    # это один и тот же объект и в процессе проявления он мог пропадать и появляться снова и снова
                    obj2.set_det_timestamp(min_det_timestamp)
                    obj2.set_dis_counter(max(obj1.disappearance_counter, obj2.disappearance_counter))
                    # а также переприсваиваем кадры оставления
                    obj2.set_leaving_frames(
                        obj1.leaving_frames if obj1.detection_timestamp == min_det_timestamp else obj2.leaving_frames)
                    return obj1
                else:
                    obj1.set_det_timestamp(min_det_timestamp)
                    obj1.set_dis_counter(max(obj1.disappearance_counter, obj2.disappearance_counter))
                    obj1.set_leaving_frames(
                        obj1.leaving_frames if obj1.detection_timestamp == min_det_timestamp else obj2.leaving_frames)
                    return obj2

        # таким образом убираем те объекты
        duplicates_tasks = [asyncio.create_task(check_pair(obj1, obj2)) for obj1, obj2
                            in combinations(self.detected_objects, 2) if obj1.suspicious and obj2.suspicious]
        duplicates = await asyncio.gather(*duplicates_tasks)
        self.detected_objects = [obj for obj in self.detected_objects if obj not in duplicates or not obj.suspicious]

    async def __update_unattended_objects(self) -> list:
        """
        Обновление оставленных предметов с возвратом списка масок, с учетом того, что данный
            оставленный предмет больше не наблюдается в маске временно статических объектов.
        :return: List из масок оставленных предметов.
        """

        async def update_object(unattended_object: UnattendedObject) -> np.array or None:
            """Обновление счетчика с возвратом маски, а также его сохранение."""
            if not unattended_object.saved:  # сохраняем, если еще не сохранен
                await save_unattended_object(unattended_object)
            # учитываем то, что данный оставленный больше не наблюдается
            if unattended_object.object_id not in [det_obj.object_id for det_obj in self.detected_objects]:
                unattended_object.update(fill_black_timeout=1)  # убавляем счетчик
                return unattended_object.contour_mask  # возвращаем маску
            else:
                return None

        # обновляем объекты и складываем маски
        update_tasks = [asyncio.create_task(update_object(unattended_object))
                        for unattended_object in self.unattended_objects]
        unattended_masks = await asyncio.gather(*update_tasks)
        # фильтруем None
        unattended_masks = [mask for mask in unattended_masks if mask is not None]
        # удаляем объекты по таймауту
        self.unattended_objects = [unattended_object for unattended_object in self.unattended_objects
                                   if unattended_object.fill_black_timeout != 0]
        return unattended_masks

    async def __plot_bboxes(self) -> None:
        """
        Отрисовка bbox'ов подозрительных или оставленных предметов.
        :return: None.
        """

        async def plot(object_data: DetectedObject) -> None:
            """Строим один bbox."""
            x1, y1, x2, y2 = object_data.bbox_coordinates
            # подозрительный объект - желтый, оставленный - красный
            color = (30, 255, 255) if not object_data.unattended else (0, 0, 255)
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)

        plot_tasks = [asyncio.create_task(plot(detected_object))
                      for detected_object in self.detected_objects
                      if detected_object.suspicious or detected_object.unattended]
        [await task for task in plot_tasks]

    async def detect_(self, current_frame: np.array) -> np.array:
        """
        Нахождение оставленных предметов в последовательности кадров.
        :param current_frame: Текущее изображение последовательности.
        :return: ДЛЯ ДЕМОНСТРАЦИИ кадры с отрисованными подозрительными (желтая рамка)
            и оставленными предметами (красная рамка).
        """
        # копим историю кадров
        self.history_frames.append(current_frame.copy())
        # обновялем оставленные и берем маски
        unattended_masks = await self.__update_unattended_objects() if self.unattended_objects else None
        # получаем bbox'ы из модели фона и удаляем из маски подтвержденные оставленные предметы, чтобы детекция не
        # сработала повторно на часть объекта, образовавшегося от распада полного на несколько составляющих
        if self.remove_people:  # плюс удаляем из маски людей, если это требуется
            detections: ultralytics.engine.results = self.yolo_seg.predict(
                current_frame, classes=[0], verbose=False, conf=self.yolo_conf)[0]
            mask_data, tso_mask = await self.tso_detector.process_frame(
                current_frame, detections.masks, unattended_masks)
        else:
            mask_data, tso_mask = await self.tso_detector.process_frame(
                current_frame, None, unattended_masks)
        # сопоставляем новые с уже имеющимися
        await self.__match_mask_data(mask_data)
        if self.detected_objects:
            # проверяем, не залежался ли какой-либо предмет + обновляем счетчик отсутствия и удаляем унесенные
            await self.__update_detected_objects()
            # удаляем возможные дубликаты подозрительных предметов
            await self.__check_suspicious_duplicates()
        self.frame = current_frame
        # отрисовываем подозрительные и/или оставленные объекты (временное решение)
        await self.__plot_bboxes()
        # return self.frame
        return np.concatenate([self.frame, cv2.merge((tso_mask, tso_mask, tso_mask))], axis=1)
