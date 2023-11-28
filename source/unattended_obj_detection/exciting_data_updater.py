import asyncio
from itertools import combinations

import cv2
import numpy as np

from utils.support_functions import iou, save_unattended_object
from utils.templates import DetectedObject, UnattendedObject


class DataUpdater:
    """Вспомогательный класс для обновления данных в базах обнаруженных и оставленных объектов."""

    def __init__(self, suspicious_timeout: int, unattended_timeout: int, disappearance_timeout: int):
        self.suspicious_timeout = suspicious_timeout
        self.unattended_timeout = unattended_timeout
        self.disappearance_timeout = disappearance_timeout
        # время, которое сегмент оставленного предмета будет заливаться черным в маске после его исчезновения
        self.fill_unattended_cont_timeout = suspicious_timeout + unattended_timeout

    async def update_detected_objects(
            self, detected_objects: list[DetectedObject], unattended_objects: list[UnattendedObject],
            timestamp: float) -> tuple[list[DetectedObject] or [], list[UnattendedObject]]:
        """
        Проверяем по таймауту время наблюдения за обнаруженными объектами и, в случае
            прохождения проверки, в зависимости от времени наблюдения, помечаем объект как
            подозрительный или оставленный и, в случае выявления последнего, добавляем к оставленным.
        Также обновляем счетчик отсутствия и удаляем те объекты, в которых счетчик достиг нуля
            (то есть предмета больше в кадре нет).
        :param detected_objects: Список обнаруженных предметов, который нужно обновить.
        :param unattended_objects: Список оставленных предметов, который нужно обновить.
        :param timestamp: Текущий timestamp.
        :return: Tuple из списков обнаруженных и оставленных предметов.
        """

        async def update_object(detected_object: DetectedObject) -> DetectedObject:
            """Обновление одного объекта."""
            # проверка по таймауту на подозрительно долгое пребывание в кадре
            obs_time = timestamp - detected_object.detection_timestamp
            if obs_time >= self.suspicious_timeout and not detected_object.suspicious and detected_object.updated:
                # помечаем его как подозрительный
                detected_object.update(suspicious=True)
            elif obs_time >= self.unattended_timeout and not detected_object.unattended and detected_object.updated:
                # по итоговому контуру из маски, находим координаты bbox'а оставленного предмета
                x, y, w, h = cv2.boundingRect(cv2.findContours(
                    np.uint8(detected_object.contour_mask.copy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0])
                # добавляем в список с оставленными с наследованием id
                unattended_objects.append(
                    UnattendedObject(
                        object_id=detected_object.object_id, contour_mask=detected_object.contour_mask,
                        bbox_coordinates=np.array([x, y, x + w, y + h]),
                        leaving_frames=detected_object.leaving_frames,
                        detection_timestamp=detected_object.detection_timestamp
                    ))
                # помечаем его как оставленный в списке обнаруженных, а также обновляем bbox
                detected_object.update(unattended=True, bbox_coordinates=np.array([x, y, x + w, y + h]))
            # смотрим, что объект не был сопоставлен в текущем кадре
            if not detected_object.updated:
                # + проверка по таймауту отсутствия => если предмет ее не проходит, он отфильтровывается
                if timestamp - detected_object.last_seen_timestamp < self.disappearance_timeout:
                    return detected_object
            else:
                detected_object.update(updated=False)  # меняем флаг на False для следующего кадра
                return detected_object

        # проверяем и обновляем обнаруженные
        update_detected_tasks = [asyncio.create_task(update_object(detected_object))
                                 for detected_object in detected_objects]
        detected_objects = await asyncio.gather(*update_detected_tasks)
        # фильтруем None
        detected_objects = [detected_object for detected_object in detected_objects if detected_object is not None]
        return detected_objects, unattended_objects

    @staticmethod
    async def check_suspicious_duplicates(detected_objects: list[DetectedObject]) -> list[DetectedObject]:
        """
        Проверка и удаление дубликатов подозрительных предметов:
            Так как крупные предметы проявляются в маске постепенно => могут дублироваться; а так как
            проверять все обнаруженные объекты не имеет смысла, ввиду того, что они появляются и пропадают
            достаточно быстро и большинство из них не сопоставится, проверяем только подозрительные.
        :param detected_objects: Список обнаруженных предметов.
        :return: Список из обнаруженных предметов без дубликатов.
        """

        async def check_pair(obj1: DetectedObject, obj2: DetectedObject) -> DetectedObject:
            """Смотрим, пересекаются ли объекты и возвращаем тот, что появился позже"""
            if iou(obj1.bbox_coordinates, obj2.bbox_coordinates) > 0:
                return obj2 if obj1.detection_timestamp < obj2.detection_timestamp else obj1

        # таким образом убираем те объекты, которые задублировались и их время обнаружения позже
        duplicates_tasks = [asyncio.create_task(check_pair(obj1, obj2)) for obj1, obj2
                            in combinations(detected_objects, 2) if obj1.suspicious and obj2.suspicious]
        duplicates = await asyncio.gather(*duplicates_tasks)
        return [obj for obj in detected_objects if obj not in duplicates or not obj.suspicious]

    async def update_unattended_objects(
            self, detected_objects: list[DetectedObject], unattended_objects: list[UnattendedObject],
            timestamp: float) -> tuple[list[UnattendedObject], list[np.array] or []]:
        """
        Обновление оставленных предметов с возвратом списка масок, с учетом того, что данный
            оставленный предмет больше не наблюдается в маске временно статических объектов.
        :param detected_objects: Список обнаруженных предметов.
        :param unattended_objects: Список оставленных предметов.
        :param timestamp: Текущий timestamp.
        :return: Tuple из списков обновленных оставленных предметов и масок оставленных предметов.
        """

        async def update_object(unattended_object: UnattendedObject) -> np.array or None:
            """Обновление счетчика с возвратом маски, а также его сохранение."""
            # ВРЕМЕННО просто сохраняем кадры
            if not unattended_object.saved:  # сохраняем, если еще не сохранен
                await save_unattended_object(unattended_object)
            # учитываем то, что данный оставленный больше не наблюдается
            if unattended_object.object_id not in [det_obj.object_id for det_obj in detected_objects]:
                unattended_object.update(obs_loss_timestamp=timestamp)
                return unattended_object.contour_mask  # возвращаем маску
            else:
                return None

        # обновляем объекты и складываем маски
        update_tasks = [asyncio.create_task(update_object(unattended_object))
                        for unattended_object in unattended_objects]
        unattended_masks = await asyncio.gather(*update_tasks)
        # фильтруем None
        unattended_masks = [mask for mask in unattended_masks if mask is not None]
        # удаляем объекты по таймауту
        unattended_objects = [unattended_object for unattended_object in unattended_objects if
                              timestamp - unattended_object.obs_loss_timestamp < self.fill_unattended_cont_timeout]
        return unattended_objects, unattended_masks
