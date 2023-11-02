"""
06.10.2023
@tarasqua
"""

import numpy as np
import cv2


class ROIPolygon:
    """Выделение области интереса (ROI - Region Of Interest)"""

    def __init__(self):
        self.polygon_points = np.empty((0, 2), dtype=int, order='C')
        self.frame_copy = None
        self.is_polygon_closed = False

    def __check_click(self, event, x, y, flags, param) -> None:
        """Слушатель нажатия кнопок"""
        if event == cv2.EVENT_LBUTTONDOWN:  # левая кнопка мыши
            if flags == 33:  # alt
                self.is_polygon_closed = True
            else:
                self.polygon_points = np.append(self.polygon_points, np.array([[x, y]]).astype(int), axis=0)
        if event == cv2.EVENT_MBUTTONDOWN:  # колесико мыши
            self.polygon_points = self.polygon_points[:-1]

    def __draw_polygon(self) -> None:
        """Отрисовка линий"""
        if self.polygon_points.shape[0] > 1:  # отрисовка прямых
            self.frame_copy = cv2.polylines(
                self.frame_copy, [self.polygon_points], self.is_polygon_closed, (60, 20, 220), 2)
        if self.polygon_points.shape[0] > 0:  # Отрисовка точек
            for point in self.polygon_points:
                cv2.circle(self.frame_copy, point, 5, (0, 200, 0), -1)

    def get_roi(self, image: np.array) -> np.array:
        """
        Возвращает точки полигона зоны интереса
        Parameters:
            image: изображение в формате np.array

        Returns:
            polygon_points: np.ndarray формата [[x, y], [x, y], ...]
        """
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.__check_click)
        while True:
            self.frame_copy = image.copy()
            self.__draw_polygon()
            cv2.imshow('image', self.frame_copy)
            if cv2.waitKey(20) & 0xFF == 27 or self.is_polygon_closed:
                break
        cv2.destroyAllWindows()
        return self.polygon_points
