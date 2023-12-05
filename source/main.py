import time
import asyncio
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np

from utils.roi_polygon_selector import ROIPolygonSelector
from utils.dist_zones_selector import DistZonesSelector
from source.unattended_obj_detection.uod import UOD


class Main:

    def __init__(self, stream_source: str | int):
        self.stream_source = stream_source

    @staticmethod
    def __get_demo(file_name: str, im_shape: tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        return cv2.VideoWriter(f'{file_name}', fourcc, 30.0, im_shape)

    def __check_click(self, event, x, y, flags, param) -> None:
        """Слушатель нажатия кнопок"""
        if event == cv2.EVENT_LBUTTONDOWN:  # левая кнопка мыши
            print(f'x={x}, y={y}')

    async def main(self):
        """Временная имплементация"""
        cap = cv2.VideoCapture(self.stream_source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        _, frame = cap.read()
        assert frame is not None, "Couldn't open stream source"
        roi = ROIPolygonSelector().get_roi(frame)
        dist_zones = DistZonesSelector().get_dist_zones(frame)
        uod = UOD(frame.shape, frame.dtype, roi, dist_zones, int(fps), False)
        # demo = self.__get_demo('demo_uod.mp4', (1440, 576))
        cv2.namedWindow('foreground')
        cv2.setMouseCallback('foreground', self.__check_click)
        while cap.isOpened():
            _, frame = cap.read()
            if frame is None:
                break
            frame = await uod.detect_(frame, time.time())
            # demo.write(frame)
            cv2.imshow('foreground', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        # demo.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # root = tk.Tk()
    # root.withdraw()
    # file_path = filedialog.askopenfilename()
    main = Main(2)
    asyncio.run(main.main())
