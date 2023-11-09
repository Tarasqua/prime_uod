import asyncio

import cv2
import numpy as np

from config_loader import Config
from roi_polygon import ROIPolygon
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
        # roi = np.array(
        #     [[279, 84], [203, 129], [131, 179], [32, 259], [14, 278], [12, 570], [705, 571], [704, 517], [703, 425],
        #      [550, 257], [451, 168], [423, 161], [379, 77]])
        uod = UOD(frame.shape, frame.dtype, roi, False)
        # demo = self.__get_demo('demo_uod.mp4', (int(cap.get(3)), int(cap.get(4))))
        # cv2.namedWindow('foreground')
        # cv2.setMouseCallback('foreground', self.__check_click)
        frames_counter = 0
        while cap.isOpened():
            _, frame = cap.read()
            if frame is None:
                break
            # if self.timer_flag:
            #     frames_counter += 1
            #     print(frames_counter)
            frame = await uod.detect_(frame)
            # demo.write(frame)
            cv2.imshow('foreground', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        # demo.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main = Main('../resources/video/video12.avi')
    # main = Main('../resources/video/video1.avi')
    # main = Main('../resources/video/video5.avi')
    # main = Main('rtsp://admin:Qwer123@192.168.9.219/cam/realmonitor?channel=1&subtype=0')
    asyncio.run(main.main())
