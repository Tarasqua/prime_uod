import asyncio
import time
from itertools import combinations

import numpy as np

from utils.support_functions import iou, save_unattended_object
from utils.templates import DetectedObject, UnattendedObject


class DataUpdater:
    """Вспомогательный класс для обновления данных в базах обнаруженных и оставленных объектов."""

    def __init__(self, suspicious_timeout: int, unattended_timeout: int):
        self.suspicious_timeout = suspicious_timeout
        self.unattended_timeout = unattended_timeout
        # время, которое сегмент оставленного предмета будет заливаться черным в маске после его исчезновения
        self.fill_unattended_cont_timeout = suspicious_timeout + unattended_timeout

    async def update_detected_objects(
            self, detected_objects: list[DetectedObject], unattended_objects: list[UnattendedObject]) -> (
            tuple)[list[DetectedObject], list[UnattendedObject]]:
        """
        Проверяем по таймауту время наблюдения за обнаруженными объектами и, в случае
            прохождения проверки, в зависимости от времени наблюдения, помечаем объект как
            подозрительный или оставленный и, в случае выявления последнего, добавляем к оставленным.
        Также обновляем счетчик отсутствия и удаляем те объекты, в которых счетчик достиг нуля
            (то есть предмета больше в кадре нет).
        :return: Tuple из списков обнаруженных и оставленных предметов.
        """

        async def update_object(detected_object: DetectedObject) -> None:
            """Обновление одного объекта."""
            # проверка по таймауту на подозрительно долгое пребывание в кадре
            obs_time = time.time() - detected_object.detection_timestamp
            if obs_time >= self.suspicious_timeout and not detected_object.suspicious:
                # помечаем его как подозрительный
                detected_object.update(suspicious=True)
            elif obs_time >= self.unattended_timeout and not detected_object.unattended:
                # добавляем в список с оставленными с наследованием id
                unattended_objects.append(
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
                                for detected_object in detected_objects]
        [await task for task in check_detected_tasks]
        # удаляем унесенные объекты
        detected_objects = [detected_object for detected_object in detected_objects
                            if detected_object.disappearance_counter != 0]
        return detected_objects, unattended_objects

    @staticmethod
    async def check_suspicious_duplicates(detected_objects: list[DetectedObject]) -> list[DetectedObject]:
        """
        Проверка и удаление дубликатов подозрительных предметов:
            Так как крупные предметы проявляются в маске постепенно => могут дублироваться; а так как
            проверять все обнаруженные объекты не имеет смысла, ввиду того, что они появляются и пропадают
            достаточно быстро и большинство из них не сопоставится, проверяем только подозрительные.
        :return: Список из обнаруженных предметов без дубликатов.
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
                            in combinations(detected_objects, 2) if obj1.suspicious and obj2.suspicious]
        duplicates = await asyncio.gather(*duplicates_tasks)
        return [obj for obj in detected_objects if obj not in duplicates or not obj.suspicious]

    async def update_unattended_objects(
            self, detected_objects: list[DetectedObject], unattended_objects: list[UnattendedObject]) -> (
            tuple[list[UnattendedObject], list[np.array] or []]):
        """
        Обновление оставленных предметов с возвратом списка масок, с учетом того, что данный
            оставленный предмет больше не наблюдается в маске временно статических объектов.
        :return: Tuple из списков обновленных оставленных предметов и масок оставленных предметов.
        """

        async def update_object(unattended_object: UnattendedObject) -> np.array or None:
            """Обновление счетчика с возвратом маски, а также его сохранение."""
            # ВРЕМЕННО просто сохраняем кадры
            if not unattended_object.saved:  # сохраняем, если еще не сохранен
                await save_unattended_object(unattended_object)
            # учитываем то, что данный оставленный больше не наблюдается
            if unattended_object.object_id not in [det_obj.object_id for det_obj in detected_objects]:
                unattended_object.update(obs_loss_timestamp=time.time())
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
        current_time = time.time()
        unattended_objects = [unattended_object for unattended_object in unattended_objects if
                              current_time - unattended_object.obs_loss_timestamp < self.fill_unattended_cont_timeout]
        return unattended_objects, unattended_masks
