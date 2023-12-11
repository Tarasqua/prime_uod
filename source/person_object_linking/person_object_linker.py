import asyncio
from typing import List, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
import ultralytics.engine.results

from utils.templates import UnattendedObject
from utils.support_functions import set_yolo_model, inflate_polygon, iou, save_unattended_object
from source.config_loader import Config


class PersObjLinker:
    """Связывание человека и предмета."""

    def __init__(self):
        config_ = Config('config.yml')
        self.similarity_threshold = config_.get(
            'PERS_OBJ_LINKER', 'SIMILARITY_THRESHOLD') / 100
        self.yolo_pose = set_yolo_model(
            config_.get('PERS_OBJ_LINKER', 'HUMAN_DETECTION', 'YOLO_MODEL'),
            'boxes', 'detect')
        self.yolo_conf = config_.get('PERS_OBJ_LINKER', 'HUMAN_DETECTION', 'YOLO_CONFIDENCE')
        self.max_inflate_bbox = config_.get('PERS_OBJ_LINKER', 'MAX_INFLATE_BBOX')
        self.inflate_step = config_.get('PERS_OBJ_LINKER', 'INFLATE_BBOX_STEP')
        self.n_clusters = config_.get('PERS_OBJ_LINKER', 'KMEANS_N_CLUSTERS')

    async def __find_leaving_frame(self, unattended_object: UnattendedObject) -> None:
        """
        Нахождение момента оставления предмета:
            двигаемся по истории кадров в обратном порядке (т.к. нам нужно будет первое вхождение с конца),
            сравнивая каждый кадр с кадром подтверждения с помощью SSIM => получая статистику по score похожести
            кадров => аппроксимируем статистику, чтобы убрать возможные резкие скачки => ищем первое вхождение
            ниже порогового (если нет, по дефолту возвращаем крайнее значение).
        На выходе имеем оставленный предмет, в котором подменяем список из всех кадров на список из одного кадра
            оставления, тем самым, не меняя формат данных.
        :param unattended_object: Данные по оставленному предмету.
        :return: None.
        """

        def struct_sim_check(bbox_coordinates: np.array) -> Tuple[int, int, int, int]:
            """
            Проверка на минимальный размер кадра для structural_similarity - 7х7 пикселей.
            :param bbox_coordinates: Координаты bbox'а предмета.
            :return: Распакованные и, если необходимо, раздутые координаты bbox'а в формате x1, y1, x2, y2.
            """
            tl_x, tl_y, br_x, br_y = bbox_coordinates
            if br_x - tl_x < 7 or br_y - tl_y < 7:
                # если координаты bbox'а меньше 7x7, раздуваем его
                inflated = inflate_polygon(
                    np.array([[tl_x, tl_y], [br_x, tl_y], [br_x, br_y], [tl_x, br_y]]),
                    np.ceil(7 / min(br_x - tl_x, br_y - tl_y)))
                return tuple(np.concatenate([inflated[0], inflated[2]]).astype(int))
            return tl_x, tl_y, br_x, br_y

        mask = np.uint8(unattended_object.contour_mask)
        # распаковываем и делаем проверку на то, что координаты bbox'а удовлетворяют structural_similarity
        x1, y1, x2, y2 = struct_sim_check(unattended_object.bbox_coordinates)
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
        unattended_object.leaving_frames = [unattended_object.leaving_frames[
            # берем первое вхождение, ниже пороговго с конца (т.к. статистика шла с конца)
            -next((i for i, s in enumerate(approximated_similarity_scores) if s <= self.similarity_threshold),
                  len(approximated_similarity_scores))]]  # если нет ниже порога, берем крайнее

    async def __nearest_object_people(
            self, detections: ultralytics.engine.results.Results,
            obj_bbox: np.array) -> List[ultralytics.engine.results.Results]:
        """
        Определение ближайших к предмету людей путем кластеризации с помощью KMeans.
        :param detections: YOLO-detections всех людей в кадре.
        :param obj_bbox: Координаты bbox'а оставленного предмета.
        :return: Список ближайших к предмету людей.
        """

        def get_bottom_centroid(detection: ultralytics.engine.results.Results) -> np.array:
            """
            Нахождение центроида по нижней грани bbox'а человека.
            :param detection: YOLO-detection.
            :return: Координаты центроида в формате np.array([x, y])
            """
            x1, y1, x2, y2 = detection.boxes.xyxy.numpy()[0]
            return np.array([(x1 + x2) / 2, y2])

        # находим центроиды последовательно, чтобы можно было однозначно сопоставить kmeans-лейбл и человека
        centroids = [get_bottom_centroid(det) for det in detections]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init='auto').fit(np.array(centroids))
        # находим лейбл предмета
        prediction = kmeans.predict(np.array([[(obj_bbox[0] + obj_bbox[2]) / 2, obj_bbox[3]]]))
        # возвращаем только тех людей, лейбл которых совпадает с лейблом предмета
        return [det for i, det in enumerate(detections) if kmeans.labels_[i] == prediction[0]]

    async def __link(self, unattended_object: UnattendedObject) -> None:
        """
        Связывание предмета с предполагаемым человеком (или несколькими людьми), который мог его оставить:
            - находим людей в кадре;
            - проверяем их количество - если их достаточно много, используем кластеризацию для фильтрации, т.к.
                в противном случае, из-за перспективных искажений, люди, стоящие близко к камере, но далеко от
                предмета, могут стать теми, кто, предположительно, оставил предмет;
            - смотрим, не пересекаются ли нижние трети bbox'ов кого-либо из людей в кадре с bbox'ом предмета;
            - если нет, раздуваем bbox до тех пор, пока не будет пересечения;
            - записываем в данный объект изображение предполагаемого человека (людей).
        :param unattended_object: Оставленный предмет (объект класса UnattendedObject).
        :return: None.
        """

        async def check_detection(detection: ultralytics.engine.results.Results, bbox: np.array) -> bool:
            """
            Проверка на то, что нижняя треть (именно та, где, скорее всего, находятся конечности) bbox'а
                данного человека пересекаются с bbox'ом предмета.
            :param detection: YOLO detection.
            :param bbox: Bbox предмета.
            :return: True, если пересекаются, иначе - False.
            """
            det_bbox = detection.boxes.xyxy.numpy()[0].copy()
            det_bbox[1] += (det_bbox[3] - det_bbox[1]) / 3
            return True if iou(bbox, det_bbox) > 0 else False

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
            unattended_object.leaving_frames[0], classes=[0], verbose=False)[0]
        # если никого не нашли
        if len(detections) == 0:
            unattended_object.set_prob_left(None)  # предполагамых оставителей в None
            return  # и брякаемся
        # если людей в кадре нашлось много, используем кластеризацию
        detections = await self.__nearest_object_people(detections, unattended_object.bbox_coordinates) \
            if len(detections) >= self.n_clusters else detections
        # сначала проверяем на пересечение с исходным bbox'ом
        scale_multiplier, prob_left = 1, []
        obj_x1, obj_y1, obj_x2, obj_y2 = unattended_object.bbox_coordinates
        polygon = np.array([[obj_x1, obj_y1], [obj_x2, obj_y1], [obj_x2, obj_y2], [obj_x1, obj_y2]])
        # ищем до первого (первых) пересечения
        while scale_multiplier <= self.max_inflate_bbox:
            polygon = inflate_polygon(polygon, scale_multiplier) if scale_multiplier != 1 else polygon
            prob_left = [
                det for det in detections if await check_detection(det, np.concatenate([polygon[0], polygon[2]]))]
            if prob_left:
                break
            scale_multiplier += self.inflate_step
        prob_left = detections if not prob_left else prob_left  # если вдруг никого не нашли, возвращаем всех
        people_frames = await asyncio.gather(*[asyncio.create_task(
            get_prob_left_frame(det, unattended_object.leaving_frames[0].copy())) for det in prob_left])
        unattended_object.set_prob_left(people_frames)

    async def link_objects(self, unattended_objects: List[UnattendedObject]) -> None:
        """
        Связывание оставленных объектов и предполагаемых людей, оставивших их:
            - находим кадр оставления предмета;
            - находим человека (или людей), которые, предположительно, могли оставить этот предмет.
        :param unattended_objects: Список оставленных предметов.
        :return: None
        """
        # находим кадр оставления предмета
        [await task for task in
         [asyncio.create_task(self.__find_leaving_frame(obj)) for obj in unattended_objects]]
        # связываем предполагаемых оставителей с ним
        [await task for task in
         [asyncio.create_task(self.__link(obj)) for obj in unattended_objects]]
        # для демонстрации сохраняем предмет и оставителя (оставителей)
        [await task for task in
         [asyncio.create_task(save_unattended_object(obj)) for obj in unattended_objects]]
