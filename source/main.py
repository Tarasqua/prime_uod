import asyncio

import cv2

from config_loader import Config
from roi_polygon import ROIPolygon
from uod import UOD


class Main:

    def __init__(self, stream_source: str):
        self.stream_source = stream_source

    @staticmethod
    def __get_demo(file_name: str, im_shape: tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        return cv2.VideoWriter(f'{file_name}', fourcc, 30.0, im_shape)

    async def main(self):
        """Временная имплементация"""
        config_ = Config('config.yml')
        cap = cv2.VideoCapture(self.stream_source)
        _, frame = cap.read()
        assert frame is not None, "Couldn't open stream source"
        roi = ROIPolygon().get_roi(frame)
        uod = UOD(frame.shape, frame.dtype, roi, False)
        # demo = self.__get_demo('demo_fg_without.mp4', (int(cap.get(3)), int(cap.get(4))))
        while cap.isOpened():
            _, frame = cap.read()
            if frame is None:
                break
            frame = await uod.detect_(frame)
            # demo.write(cv2.merge((frame, frame, frame)))
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
