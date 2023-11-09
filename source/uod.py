"""
@tarasqua
"""
import os
import time
from pathlib import Path
import asyncio
from typing import List
from collections import deque

import cv2
import numpy as np
import ultralytics.engine.results
from ultralytics import YOLO

from config_loader import Config
from background_subtractor import BackgroundSubtractor
from templates import SuspiciousObject, UnattendedObject


class UOD:
    """
    Unattended Object Detector
    """

    def __init__(self, frame_shape: np.array, frame_dtype: np.dtype, roi: list, remove_people: bool = True):
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
        self.bg_subtractor = BackgroundSubtractor(frame_shape, frame_dtype, config_.get('BG_SUBTRACTION'), roi)
        self.yolo_seg = self.__set_yolo_model(config_.get('UOD', 'HUMAN_DETECTION', 'YOLO_MODEL')) \
            if remove_people else None
        self.yolo_conf = config_.get('UOD', 'HUMAN_DETECTION', 'YOLO_CONFIDENCE') \
            if remove_people else None
        self.area_threshold = (np.prod(np.array(frame_shape[:-1])) *  # % от площади кадра
                               (config_.get('UOD', 'AREA_THRESH_FRAME_PERCENT') * 0.01))
        self.con_comp_connectivity = config_.get('UOD', 'CON_COMPONENTS_CONNECTIVITY')

        self.suspicious_objects: List[SuspiciousObject] = []  # подозрительные предметы
        # пороговое значение (в %) изменения площади для изменения координат рамки
        self.susp_obj_area_threshold = config_.get('UOD', 'SUSP_OBJ_AREA_THRESHOLD')
        self.unattended_objects: List[UnattendedObject] = []  # оставленные предметы
        # сколько кадров должен пролежать предмет для подтверждения того, что он - оставленный
        self.unattended_frames_timeout = config_.get('UOD', 'SUSPICIOUS_TO_UNATTENDED_TIMEOUT')
        # история кадров
        self.history_frames = deque(maxlen=self.unattended_frames_timeout)
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

    async def __get_mask_data(self, tso_mask: cv2.typing.MatLike) -> np.array:
        """
        Нахождение и фильтрация по площади центроидов и bbox'ов временно статических объектов из маски.
        :param tso_mask: Маска со временно статическими объектами в ней.
        :return: Массив вида np.array([[centroid_x, centroid_y, x1, y1, x2, y2], [...], ...])
        """
        # находим связные области в маске
        connected_areas = cv2.connectedComponentsWithStats(tso_mask, self.con_comp_connectivity, cv2.CV_32S)
        # обрабатываем полученные области и возвращаем их
        process_area_tasks = [asyncio.create_task(self.__process_area(centroid, stat))
                              for centroid, stat in zip(connected_areas[3][1:], connected_areas[2][1:])]
        centroids_bboxes = await asyncio.gather(*process_area_tasks)
        return np.array([bbox for bbox in centroids_bboxes if bbox is not None])  # фильтруем None

    async def __update_suspicious_object(self, suspicious_object: SuspiciousObject, data: np.array) -> None:
        """
        Обновление подозрительного объекта, в зависимости от изменения площади контура.
            Если площадь контура увеличилась более чем на n%, обновляем координаты рамки объекта.
        :param suspicious_object: Объект класса подозрительного объекта.
        :param data: Новые данные в формате np.array([centroid_x, centroid_y, x1, y1, x2, y2, area]).
        :return: None.
        """
        if (abs(suspicious_object.contour_area - data[-1]) / suspicious_object.contour_area
                > self.susp_obj_area_threshold):
            suspicious_object.update(observation_counter=1, contour_area=data[-1],
                                     bbox_coordinated=data[2:-1], centroid_coordinates=data[:2])
        else:
            suspicious_object.update(observation_counter=1)

    async def __match_new_object(self, new_obj_data: np.array) -> None:
        """
        Сопоставление нового полученного объекта с уже имеющимися в базе подозрительных
            по координатам центроидов.
        :param new_obj_data: Данные по объекту из маски в формате
            np.array([centroid_x, centroid_y, x1, y1, x2, y2, area])
        :return: None.
        """
        matched = False  # для отслеживания того, что объект сопоставился
        for suspicious_object in self.suspicious_objects:
            x1, y1, x2, y2 = suspicious_object.bbox_coordinates
            # если координаты центроида внутри bbox'а => новый объект сопоставлен => обновляем текущий
            if x1 <= new_obj_data[0] <= x2 and y1 <= new_obj_data[1] <= y2:
                await self.__update_suspicious_object(suspicious_object, new_obj_data)
                matched = True  # сопоставился
            else:
                # также отслеживаем тот факт, что данного объекта из базы нет на текущем кадре
                suspicious_object.update(updated=False)
        # если объект не был сопоставлен ни с одним из базы => это новый объект => добавляем в базу
        if not matched:
            self.suspicious_objects.append(
                SuspiciousObject(contour_area=new_obj_data[-1], bbox_coordinates=new_obj_data[2:-1],
                                 centroid_coordinates=new_obj_data[:2]))

    async def __check_suspicious(self, suspicious_object: SuspiciousObject) -> None:
        """
        Вспомогательная функция для __check_suspicious_objects -
            обработка одного подозрительного объекта.
        :param suspicious_object: Подозрительный объект.
        :return: None.
        """
        # проверка по таймауту
        if suspicious_object.observation_counter >= self.unattended_frames_timeout and \
                not suspicious_object.unattended:
            # добавляем в список с оставленными
            self.unattended_objects.append(
                UnattendedObject(bbox_coordinates=suspicious_object.bbox_coordinates,
                                 detection_frame=self.history_frames[0]))
            # помечаем его как оставленный в списке подозрительных
            suspicious_object.update(unattended=True)
        # обновление счетчика отсутствия (убавляем, если объект не был найден в текущем кадре)
        if not suspicious_object.updated:
            suspicious_object.update(disappearance_counter=1, updated=True)

    async def __check_suspicious_objects(self) -> None:
        """
        Проверяем по таймауту время наблюдения за подозрительными объектами и, в случае
            прохождения проверки, помечаем объект как оставленный и добавляем к оставленным.
        Также обновляем счетчик отсутствия и удаляем те объекты, в которых счетчик достиг нуля
            (то есть предмета больше в кадре нет).
        :return: None.
        """
        # проверяем и обновляем подозрительные
        check_suspicious_tasks = [asyncio.create_task(self.__check_suspicious(suspicious_object))
                                  for suspicious_object in self.suspicious_objects]
        [await task for task in check_suspicious_tasks]
        # удаляем унесенные объекты
        self.suspicious_objects = [suspicious_object for suspicious_object in self.suspicious_objects
                                   if suspicious_object.disappearance_counter != 0]

    async def __match_mask_data(self, mask_data: np.array) -> None:
        """
        Сопоставление только что полученных данных из маски, а также обновление уже существующих в базе данных.
        :param mask_data: Все данные, полученные из маски в формате
            np.array([[centroid_x, centroid_y, x1, y1, x2, y2, area], [...]]).
        :return: None.
        """
        if not self.suspicious_objects:  # если список пустой, добавляем все объекты
            [self.suspicious_objects.append(
                SuspiciousObject(contour_area=data[-1], bbox_coordinates=data[2:-1], centroid_coordinates=data[:2]))
                for data in mask_data]
        else:  # если не пустой
            # и если временно статических объектов в кадре не найдено
            if mask_data.size == 0:
                # выставляем флаг обновления на текущем кадре у всех объектов в базе на False
                for suspicious_object in self.suspicious_objects:
                    suspicious_object.update(updated=False)
            else:  # если же в кадре есть временно статические объекты
                # связываем только что полученные с уже имеющимися объектами
                match_bbox_tasks = [asyncio.create_task(self.__match_new_object(data)) for data in mask_data]
                [await task for task in match_bbox_tasks]
            # проверяем, не залежался ли какой-либо предмет + обновляем счетчик отсутствия и удаляем унесенные
            await self.__check_suspicious_objects()

    async def __save_unattended_object(self, obj_data: UnattendedObject):
        x1, y1, x2, y2 = obj_data.bbox_coordinates
        cv2.rectangle(obj_data.detection_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        detections_path = os.path.join(Path(__file__).resolve().parents[1], 'resources', 'uod_detections')
        im_path = os.path.join(detections_path, f'{len(os.listdir(detections_path)) + 1}.png')
        cv2.imwrite(im_path, obj_data.detection_frame)

    async def plot_(self, object_data: SuspiciousObject) -> None:
        """
        Отрисовка bbox'ов и id объектов.
        :param object_data: Данные по объекту - объект класса SuspiciousObject.
        :return: None.
        """
        x1, y1, x2, y2 = object_data.bbox_coordinates
        # подозрительный объект - желтый, оставленный - красный
        color = (30, 255, 255) if not object_data.unattended else (0, 0, 255)
        cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)

    async def detect_(self, current_frame: np.array) -> np.array:
        """
        Нахождение оставленных предметов в последовательности кадров.
        :param current_frame: Текущее изображение последовательности.
        :return: ДЛЯ ДЕМОНСТРАЦИИ кадры с отрисованными подозрительными (желтая рамка)
            и оставленными предметами (красная рамка).
        """
        # копим историю кадров
        self.history_frames.append(current_frame)
        # получаем bbox'ы из модели фона
        if self.remove_people:
            detections: ultralytics.engine.results = self.yolo_seg.predict(
                current_frame, classes=[0], verbose=False, conf=self.yolo_conf)[0]
            tso_mask = await self.bg_subtractor.get_tso_mask(current_frame, detections.masks)
        else:
            tso_mask = await self.bg_subtractor.get_tso_mask(current_frame, None)
        mask_data = await self.__get_mask_data(tso_mask)
        await self.__match_mask_data(mask_data)
        self.frame = current_frame
        # строим подозрительные объекты (временное решение)
        plot_tasks = [asyncio.create_task(self.plot_(data)) for data in self.suspicious_objects]
        [await task for task in plot_tasks]
        # сохраняем найденные оставленные предметы
        if self.unattended_objects:
            save_tasks = [asyncio.create_task(self.__save_unattended_object(obj)) for obj in self.unattended_objects]
            [await task for task in save_tasks]
            self.unattended_objects.clear()
        return self.frame
