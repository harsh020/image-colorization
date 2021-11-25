from colorizer.models import models


class ModelFactory:
    MODELS = {
        'colornet18': models.ColorNet18,
        'constantnet': models.ConstantNet
    }

    def __init__(self):
        pass

    @classmethod
    def get(cls, name):
        if name not in list(cls.MODELS.keys()):
            raise ValueError(f'Invalid value {name} for model name. \
                               Should be on of {lsit(cls.MODELS.keys())}')

        return cls.MODELS[name]


    @classmethod
    def list(cls):
        return list(cls.MODELS.keys())
