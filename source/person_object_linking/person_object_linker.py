import asyncio
from typing import List

import cv2
import numpy as np
import torch

from utils.templates import UnattendedObject
from source.config_loader import Config


class PersObjLinker:
    """Связывание человека и предмета."""
    def __init__(self):
        config_ = Config('config.yml')
        self.dif_ref_similarity_threshold = config_.get(
            'PERS_OBJ_LINKER', 'SIMILARITY_THRESHOLD') / 100

    def find_leaving_frame(self, uo_data: UnattendedObject) -> None:
        """
        Нахождение момента оставления предмета:
            - находим площадь сегмента оставленного предмета и заливаем вне контура объекта черным;
            - отматываемся по истории обратно, находим разницу между подтвержденным кадром и кадром
                истории, бинаризуя разницу, чтобы можно было точно сравнить с маской;
            - если площади исходной маски и площадь разницы сильно отличаются то считаем,
                что данный кадр - момент оставления.
        :param uo_data: Данные по оставленному предмету.
        :return: Кадр с моментом оставления предмета.
        """
        mask = np.uint8(uo_data.contour_mask)
        ref_area = len(mask[mask != 0])
        masked_reference_frame = cv2.bitwise_and(
            uo_data.confirmation_frame, uo_data.confirmation_frame, mask=mask)

        def compare_frames(to_compare: np.array) -> float:
            diff = cv2.absdiff(masked_reference_frame, cv2.bitwise_and(to_compare, to_compare, mask=mask))
            bin_diff = cv2.threshold(
                cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            return (ref_area - len(bin_diff[bin_diff != 0])) / ref_area

        frame_counter = 1
        while compare_frames(uo_data.leaving_frames[-frame_counter]) > self.dif_ref_similarity_threshold:
            frame_counter += 1

        uo_data.leaving_frames = uo_data.leaving_frames[0]

    def link_object(self, unattended_objects: List[UnattendedObject]):
        pass


if __name__ == '__main__':
    data = torch.load('unattended_object_data_s2v4.pt')
    pol = PersObjLinker()
    pol.find_leaving_frame(data)
