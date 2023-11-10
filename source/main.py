import asyncio
import tkinter as tk
from tkinter import filedialog

import cv2

from utils.roi_polygon import ROIPolygon
from uod import UOD


class Main:

    def __init__(self, stream_source: str):
        self.stream_source = stream_source
        self.timer_flag = False

    @staticmethod
    def __get_demo(file_name: str, im_shape: tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        return cv2.VideoWriter(f'{file_name}', fourcc, 30.0, im_shape)

    def __check_click(self, event, x, y, flags, param) -> None:
        """Слушатель нажатия кнопок"""
        if event == cv2.EVENT_LBUTTONDOWN:  # левая кнопка мыши
            self.timer_flag = not self.timer_flag

    async def main(self):
        """Временная имплементация"""
        cap = cv2.VideoCapture(self.stream_source)
        _, frame = cap.read()
        assert frame is not None, "Couldn't open stream source"
        roi = ROIPolygon().get_roi(frame)
        uod = UOD(frame.shape, frame.dtype, roi, False)
        # demo = self.__get_demo('demo_uod.mp4', (int(cap.get(3)), int(cap.get(4))))
        cv2.namedWindow('foreground')
        cv2.setMouseCallback('foreground', self.__check_click)
        counter = 1
        while cap.isOpened():
            _, frame = cap.read()
            if frame is None:
                break
            if self.timer_flag:
                print(counter)
                counter += 1
            frame = await uod.detect_(frame)
            # demo.write(frame)
            cv2.imshow('foreground', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        # demo.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    main = Main(file_path)
    asyncio.run(main.main())
