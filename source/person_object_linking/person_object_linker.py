import asyncio
import time
from typing import List

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity
from scipy.signal import savgol_filter
import ultralytics.engine.results

from utils.templates import UnattendedObject
from utils.support_functions import set_yolo_model, inflate_polygon
from source.config_loader import Config


class PersObjLinker:
    """Связывание человека и предмета."""

    def __init__(self):
        config_ = Config('config.yml')
        self.similarity_threshold = config_.get(
            'PERS_OBJ_LINKER', 'SIMILARITY_THRESHOLD') / 100
        self.yolo_pose = set_yolo_model(
            config_.get('PERS_OBJ_LINKER', 'HUMAN_DETECTION', 'YOLO_MODEL'), 'pose')
        self.yolo_conf = config_.get('PERS_OBJ_LINKER', 'HUMAN_DETECTION', 'YOLO_CONFIDENCE')
        self.max_inflate_bbox = config_.get('PERS_OBJ_LINKER', 'MAX_INFLATE_BBOX')
        self.inflate_step = config_.get('PERS_OBJ_LINKER', 'INFLATE_BBOX_STEP')

    async def __find_leaving_frame(self, unattended_object: UnattendedObject) -> None:
        """
        Нахождение момента оставления предмета:
            двигаемся по истории кадров в обратном порядке (т.к. нам нужно будет первое вхождение с конца),
            сравнивая каждый кадр с кадром подтверждения с помощью SSIM => получая статистику по score похожести
            кадров => аппроксимируем статистику, чтобы убрать возможные резкие скачки => ищем первое вхождение
            ниже порогового (если нет, по дефолту возвращаем крайнее значение).
        :param unattended_object: Данные по оставленному предмету.
        :return: Кадр с моментом оставления предмета.
        """
        mask = np.uint8(unattended_object.contour_mask)
        x1, y1, x2, y2 = unattended_object.bbox_coordinates
        # мАскируем, грейскейлим и обрезаем по ббоксу кадр подтверждения
        masked_reference_frame = cv2.cvtColor(cv2.bitwise_and(
            unattended_object.confirmation_frame, unattended_object.confirmation_frame, mask=mask),
            cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2]
        # собираем по кадровую статистику схожести с кадром подтверждения
        similarity_scores = [structural_similarity(
            masked_reference_frame,
            # мАскируем, грейскейлим и обрезаем по ббоксу кадр подтверждения
            cv2.cvtColor(cv2.bitwise_and(frame, frame, mask=mask), cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2],
            full=True)[0]  # берем только score, без маски
                             for frame in unattended_object.leaving_frames[::-1]]  # бежим в обратном порядке
        approximated_similarity_scores = savgol_filter(  # аппроксимируем с помощью фильтра Савицкого-Голея
            similarity_scores, int(len(unattended_object.leaving_frames) / 20), 1)
        # подменяем в конкретном экземпляре кадр с оставлением
        unattended_object.leaving_frames = unattended_object.leaving_frames[
            # берем первое вхождение, ниже пороговго с конца (т.к. статистика шла с конца)
            -next((i for i, s in enumerate(approximated_similarity_scores) if s <= self.similarity_threshold),
                  len(approximated_similarity_scores))]  # если нет ниже порога, берем крайнее

    async def __link(self, unattended_object: UnattendedObject):
        """
        Связывание предмета с предполагаемым человеком (или несколькими людьми), который мог его оставить:
            - находим людей в кадре;
            - смотрим, не пересекаются ли конечности кого-либо из людей в кадре с bbox'ом предмета;
            - если нет, раздуваем bbox до тех пор, пока не будет пересечения;
            - записываем в данный объект изображение предполагаемого человека (людей).
        :param unattended_object: Оставленный предмет (объект класса UnattendedObject).
        :return:
        """

        async def check_inside(point: np.array, conf: np.float32, bbox: np.array) -> bool:
            """
            Проверка на то, что точка лежит внутри bbox'а, а также conf больше порогового.
            :param point: Координаты точки: np.array([x, y]).
            :param conf: Confidence данной точки.
            :param bbox: Координаты bbox'а, в котором должна лежать точка.
            :return: True, если точка лежит внутри или на границе, а иначе - False.
            """
            return True \
                if bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3] and conf >= self.yolo_conf \
                else False

        async def check_detection(detection: ultralytics.engine.results.Results, bbox: np.array) -> bool:
            """
            Проверка на то, что конечности данного человека пересекаются с bbox'ом предмета.
            :param detection: YOLO detection.
            :param bbox: Bbox предмета.
            :return: True, если пересекаются, иначе - False.
            """
            keypoints = detection.keypoints.data.numpy()[0]
            check_tasks = [asyncio.create_task(check_inside(limb[:-1], limb[-1], bbox))
                           for limb in [keypoints[9], keypoints[10],  # кисти
                                        keypoints[15], keypoints[16]]]  # лодыжки
            checked_limbs = await asyncio.gather(*check_tasks)
            return True if any(checked_limbs) else False

        async def get_prob_left_frame(detection: ultralytics.engine.results.Results, ref_frame: np.array) -> np.array:
            """
            Возвращает изображение человека, вырезанное из исходного изображения, сделанного в момент оставления.
            :param detection: YOLO-detection человека.
            :param ref_frame: Изображение, сделанное в момент оставления.
            :return: Обрезанное изображение человека.
            """
            x1, y1, x2, y2 = detection.boxes.xyxy.numpy()[0].astype(int)
            return ref_frame[y1:y2, x1:x2]

        # находим людей в кадре
        detections: ultralytics.engine.results.Results = self.yolo_pose.predict(
            unattended_object.leaving_frames, classes=[0], verbose=False, conf=self.yolo_conf)[0]
        # сначала проверяем на пересечение с исходным bbox'ом
        scale_multiplier = 1
        obj_x1, obj_y1, obj_x2, obj_y2 = unattended_object.bbox_coordinates
        polygon = np.array([[obj_x1, obj_y1], [obj_x2, obj_y1], [obj_x2, obj_y2], [obj_x1, obj_y2]])
        prob_left = []
        # ищем до первого (первых) пересечения
        while scale_multiplier <= self.max_inflate_bbox:
            polygon = inflate_polygon(
                polygon, scale_multiplier) if scale_multiplier != 1 else polygon
            prob_left = [
                det for det in detections if await check_detection(det, np.concatenate([polygon[0], polygon[2]]))]
            if prob_left:
                break
            scale_multiplier += self.inflate_step
        prob_left = detections if not prob_left else prob_left  # если вдруг никого не нашли, возвращаем всех
        frames_tasks = [asyncio.create_task(
            get_prob_left_frame(det, unattended_object.leaving_frames.copy())) for det in prob_left]
        people_frames = await asyncio.gather(*frames_tasks)
        unattended_object.set_prob_left(people_frames)

    async def link_objects(self, unattended_objects: List[UnattendedObject]) -> List[UnattendedObject]:
        """
        Связывание оставленных объектов и предполагаемых людей, оставивших их:
            - находим кадр оставления предмета;
            - находим человека (или людей), которые, предположительно, могли оставить этот предмет.
        :param unattended_objects: Список оставленных предметов.
        :return:
        """
        [await task for task in
         [asyncio.create_task(self.__find_leaving_frame(obj)) for obj in unattended_objects]]
        [await task for task in
         [asyncio.create_task(self.__link(obj)) for obj in unattended_objects]]
        return unattended_objects


async def run():
    data: UnattendedObject = torch.load('../../resources/leaving_frame_stat/unattended_object_data_s2v4.pt')
    data.leaving_frames = data.leaving_frames[::2]
    pol = PersObjLinker()
    start = time.time()
    objects = await pol.link_objects([data])
    print(time.time() - start)
    cv2.imwrite('test.png', objects[0].probably_left_object_people[0])


if __name__ == '__main__':
    asyncio.run(run())
