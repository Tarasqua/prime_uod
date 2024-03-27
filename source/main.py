import time
import asyncio
import tkinter as tk
from tkinter import filedialog

import cv2

from utils.roi_polygon_selector import ROIPolygonSelector
from utils.dist_zones_selector import DistZonesSelector
from source.unattended_obj_detection.uod import UOD
from utils.support_functions import plot_bboxes
from source.person_object_linking.person_object_linker import PersObjLinker


def get_writer(file_name: str, im_shape: tuple[int, int]) -> cv2.VideoWriter:
    """
    Запись демо работы детектора.
    :param file_name: Наименование файла.
    :param im_shape: Разрешение входного видеопотока.
    :return: Объект класса VideWriter.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(f'{file_name}', fourcc, 30.0, im_shape)


async def main(stream_source):
    """
    Запуск детектора
    :param stream_source:
    :return:
    """
    cap = cv2.VideoCapture(stream_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    _, frame = cap.read()
    assert frame is not None, "Couldn't open stream source"
    # writer = get_writer('office UOD demo.mp4', frame.shape[:-1][::-1])
    roi = ROIPolygonSelector().get_roi(frame)
    dist_zones = DistZonesSelector().get_dist_zones(frame)
    uod = UOD(frame.shape, frame.dtype, roi, dist_zones, int(fps), False)
    pers_obj_linker = PersObjLinker()
    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break
        # возвращаем не только оставленные, но и все задетектированные предметы тоже для демонстрации
        detected_objects, unattended_objects = await uod.detect_(frame, time.time())
        # связываем предметы с предполагаемыми оставителями, если они еще не связаны
        if not_linked_unattended := [obj for obj in unattended_objects if not obj.linked]:
            await pers_obj_linker.link_objects(not_linked_unattended)
        detections_frame = await plot_bboxes(detected_objects, frame.copy())
        # writer.write(detections_frame)
        cv2.imshow('demonstration', detections_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    asyncio.run(main(file_path))
