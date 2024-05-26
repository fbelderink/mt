import yaml
import torch
from utils.hyperparameters import Hyperparameters

from datetime import datetime
from pathlib import Path
class ConfigLoader:
    def __init__(self, config_file):
        self.config_file = config_file
        with open(self.config_file, 'r', encoding='utf-8') as f:
            try:
                self.yaml = yaml.safe_load(f)
            except yaml.YAMLError as exception:
                print(exception)
                print("Couldn't load yaml file please check syntax")
                quit()

    def get_config(self):
        return self.yaml


#print(list(ConfigLoader("config.yaml").get_config()["dimensions"].values()))

