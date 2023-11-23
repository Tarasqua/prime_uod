"""
06.10.2023
@tarasqua
"""

import numpy as np
import cv2


class ROIPolygonSelector:
    """Выделение области интереса (ROI - Region Of Interest)"""

    def __init__(self):
        self.polygons = []
        self.polygon_points = np.empty((0, 2), dtype=int, order='C')
        self.frame_copy = None

    def __check_click(self, event, x, y, flags, param) -> None:
        """
        Слушатель нажатия кнопок
        :param event: Нажатая кнопка мыши.
        :param x: Координата x.
        :param y: Координата y.
        :param flags: Нажатая кнопка на клавиатуре.
        :param param: _.
        :return: None.
        """
        if event == cv2.EVENT_LBUTTONDOWN:  # левая кнопка мыши
            if flags == 33:  # alt
                self.polygons.append(self.polygon_points)
                self.polygon_points = np.empty((0, 2), dtype=int, order='C')
            else:
                self.polygon_points = np.append(self.polygon_points, np.array([[x, y]]).astype(int), axis=0)
        if event == cv2.EVENT_MBUTTONDOWN:  # колесико мыши
            self.polygon_points = self.polygon_points[:-1]

    def __draw_polygon(self) -> None:
        """
        Отрисовка текущих полигонов и точек.
        :return: None
        """
        for polygon in self.polygons:  # отрисовка уже законченных полигонов
            self.frame_copy = cv2.polylines(
                self.frame_copy, [polygon], True, (158, 159, 66), 2
            )
        if self.polygon_points.shape[0] > 1:  # отрисовка текущего полигона
            self.frame_copy = cv2.polylines(
                self.frame_copy, [self.polygon_points], False, (114, 120, 0), 2)
        if self.polygon_points.shape[0] > 0:  # отрисовка точек текущего полигона
            for point in self.polygon_points:
                cv2.circle(self.frame_copy, point, 5, (59, 95, 240), -1)

    def get_roi(self, image: np.array) -> list:
        """
        Определение нескольких ROI-полигонов.
        :param image: Изображение в формате np.array.
        :return: list из np.array формата [[x, y], [x, y], ...].
        """
        cv2.namedWindow('ROI')
        cv2.setMouseCallback('ROI', self.__check_click)
        while True:
            self.frame_copy = image.copy()
            self.__draw_polygon()
            cv2.imshow('ROI', self.frame_copy)
            if cv2.waitKey(33) == 13:  # enter, чтобы закончить
                break
        cv2.destroyAllWindows()
        return self.polygons
