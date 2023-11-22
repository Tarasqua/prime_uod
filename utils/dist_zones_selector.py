import numpy as np
import cv2


class DistZonesSelector:

    def __init__(self):
        self.dist_zones_points = []
        self.current_points = np.empty((0, 2), dtype=int, order='C')
        self.colors = [(158, 159, 66), (98, 159, 66)]
        self.frame_copy = None

    def __check_click(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:  # левая кнопка мыши
            if len(self.dist_zones_points) < 2:  # выделяем 2 зоны
                self.current_points = np.append(self.current_points, np.array([[x, y]]).astype(int), axis=0)
        if event == cv2.EVENT_MBUTTONDOWN:  # колесико мыши
            self.current_points = self.current_points[:-1]

    def __draw_dist_zones(self) -> None:
        for i, zone_points in enumerate(self.dist_zones_points):  # отрисовка уже законченных зон
            cv2.rectangle(self.frame_copy, zone_points[0], zone_points[1], self.colors[i], 2)
        if self.current_points.size > 0:  # отрисовка точек текущей зоны
            for point in self.current_points:
                cv2.circle(self.frame_copy, point, 5, (59, 95, 240), -1)

    def get_dist_zones(self, image: np.array) -> list:
        """
        Определение нескольких ROI-полигонов.
        :param image: Изображение в формате np.array.
        :return: list из np.array формата [[x, y], [x, y], ...].
        """
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.__check_click)
        while True:
            if self.current_points.size == 4:
                self.dist_zones_points.append(self.current_points)
                self.current_points = np.empty((0, 2), dtype=int, order='C')
            self.frame_copy = image.copy()
            self.__draw_dist_zones()
            cv2.imshow('image', self.frame_copy)
            if cv2.waitKey(33) == 13:  # enter, чтобы закончить
                break
        cv2.destroyAllWindows()
        return self.dist_zones_points
