import asyncio
import time

import cv2
import numpy as np

from utils.dist_zones_selector import DistZonesSelector


class DistZonesHandler:

    def __init__(self, frame_shape: np.array, dist_zones_points: np.array):
        self.frame_shape = frame_shape
        self.x_round_margin = frame_shape[1] * 0.05  # 5% для округления координат зоны по x
        self.y_round_margin = frame_shape[0] * 0.05  # и 5% по y
        # проверяем, что координаты зон в формате (top left, bottom right), а также округляем до границ кадра
        self.dist_zones_points = [np.concatenate(self.__check_points(zone_points))
                                  for zone_points in dist_zones_points]

    def __check_points(self, zone_points: np.array) -> np.array:
        """
        Перевод координат зоны в формат (top left, bottom right),
            а также округление координат зоны до границ кадра, если они близки к границам кадра.
        :param zone_points: Координаты зоны в формате np.array([[x, y], [x, y]]).
        :return: Обновленные координаты зоны в том же формате.
        """
        zone_points = np.array([[min(zone_points[:, 0]), min(zone_points[:, 1])],
                                [max(zone_points[:, 0]), max(zone_points[:, 1])]])
        if zone_points[0][0] <= self.x_round_margin:
            zone_points[0][0] = 0
        if zone_points[0][1] <= self.y_round_margin:
            zone_points[0][1] = 0
        if self.frame_shape[1] - zone_points[1][0] <= self.x_round_margin:
            zone_points[1][0] = self.frame_shape[1]
        if self.frame_shape[0] - zone_points[0][1] <= self.y_round_margin:
            zone_points[1][1] = self.frame_shape[0]
        return zone_points

    def get_dist_zones_frames(self, frame: np.array) -> list:
        """
        Получение кадров с зонами дальности, где первая - самая дальняя зона,
            а последняя - исходное изображение.
        :param frame: Исходное изображение.
        :return: List из зон дальности.
        """
        return [*[frame.copy()[zone[1]:zone[3], zone[0]:zone[2]] for zone in self.dist_zones_points], frame]


async def main(path: str):
    # ВРЕМЕННО ДЛЯ ОТЛАДКИ
    cap = cv2.VideoCapture(path)
    _, frame = cap.read()
    dist_zones = DistZonesSelector().get_dist_zones(frame)
    dzh = DistZonesHandler(frame.shape, dist_zones)
    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break
        frame1, frame2, frame3 = dzh.get_dist_zones_frames(frame)
        cv2.imshow('frame1', frame1)
        cv2.imshow('frame2', frame2)
        cv2.imshow('frame3', frame3)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import tkinter as tk
    import tkinter.filedialog as filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    asyncio.run(main(file_path))
