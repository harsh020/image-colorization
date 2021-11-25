from colorizer.losses import losses


class LossFactory:
    LOSSES = {
        'mse': losses.mse_loss(),
        'huber': losses.huber_loss(),
        'perceptual': losses.perceptual_loss(),
        'contrastive': losses.contrastive_loss()
    }

    @classmethod
    def get(cls, name):
        if name not in list(cls.LOSSES.keys()):
            raise ValueError(f'Invalid value {name} for model name. \
                               Should be on of {lsit(cls.LOSSES.keys())}')

        return cls.LOSSES[name]


    @classmethod
    def list(cls):
        return list(cls.LOSSES.keys())
