import asyncio
import time
from typing import List

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity
from scipy.signal import savgol_filter

from utils.templates import UnattendedObject
from source.config_loader import Config


class PersObjLinker:
    """Связывание человека и предмета."""

    def __init__(self):
        config_ = Config('config.yml')
        self.similarity_threshold = config_.get(
            'PERS_OBJ_LINKER', 'SIMILARITY_THRESHOLD') / 100

    async def find_leaving_frame(self, unattended_object: UnattendedObject) -> None:
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

    async def link_objects(self, unattended_objects: List[UnattendedObject]):
        [await task for task in
         [asyncio.create_task(self.find_leaving_frame(obj)) for obj in unattended_objects]]


if __name__ == '__main__':
    data: UnattendedObject = torch.load('unattended_object_data_s2v4.pt')
    data.leaving_frames = data.leaving_frames[::2]
    pol = PersObjLinker()
    start = time.time()
    leaving_frame = pol.find_leaving_frame(data)
    print(time.time() - start)
    cv2.imwrite('test.png', leaving_frame)
