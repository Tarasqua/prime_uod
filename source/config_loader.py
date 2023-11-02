"""
@tarasqua
"""

import os
import yaml
from functools import reduce
from pathlib import Path


class Config:
    """Подгрузка конфига"""

    def __init__(self, config_file_name: str):
        """
        Подгрузчик конфига.
        :param config_file_name: Название конфиг-файла в папке директории с конфигами.
        """
        config_path = os.path.join(Path(__file__).resolve().parents[1], 'resources', 'config', config_file_name)
        assert os.path.exists(config_path), "Config not found"
        with open(config_path, 'r') as f:
            self.config_loader: dict = yaml.load(f, Loader=yaml.FullLoader)

    def get(self, *setting_name):
        """
        Геттер конфига.
        :param setting_name: Путь до определенного параметра в конфиг-файле вида *args.
        :return: Запрпашиваемый параметр из конфига.
        """
        return reduce(dict.get, setting_name, self.config_loader)
