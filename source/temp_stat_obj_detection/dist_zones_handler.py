import asyncio

import numpy as np


class DistZonesHandler:
    """Обработка зон дальности."""

    def __init__(self, frame_shape: np.array, dist_zones_points: list):
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
        :return: Обновленные координаты зоны в формате np.array([x1, y1, x2, y2]).
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
        return zone_points.astype(int)

    def get_frames_shapes(self) -> list[tuple[int, int, int]]:
        """
        Получение размеров кадров с зонами дальности, где первый элемент - размер самой дальней зоны,
            а последний - размер исходного изображения.
        :return:
        """
        return [*[(points[3] - points[1], points[2] - points[0], 3)
                  for points in self.dist_zones_points], self.frame_shape]

    async def get_dist_zones_frames(self, frame: np.array) -> list[np.array]:
        """
        Получение кадров с зонами дальности, где первая - самая дальняя зона,
            а последняя - исходное изображение.
        :param frame: Исходное изображение.
        :return: List из зон дальности.
        """

        async def get_zone(zone_points: np.array) -> np.array:
            """Вспомогательная функция для выделения одной зоны дальности."""
            return frame.copy()[zone_points[1]:zone_points[3], zone_points[0]:zone_points[2]]

        zones = await asyncio.gather(*[
            asyncio.create_task(get_zone(zone)) for zone in self.dist_zones_points])
        return [*zones, frame]

    async def get_merged_mask(self, masks: list[np.array]) -> np.array:
        """
        Объединение масок зон дальности со временно статическими объектами в одну,
            в исходном разрешении кадра.
        :param masks: Список масок зон дальности со временно статическими объектами,
            где последняя - маска с общим планом.
        :return: Объединенная маска со временно статическими объектами.
        """
        async def merge_masks(mask: np.array, points: list) -> np.array:
            """Вспомогательная функция для объединения маски зоны дальности с маской общего плана."""
            masks[-1][points[1]:points[3], points[0]:points[2]] = mask

        [await task for task in [
            asyncio.create_task(merge_masks(mask, points))
            for mask, points in zip(masks[:-1][::-1], self.dist_zones_points[::-1])]]
        return masks[-1]
