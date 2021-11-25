from colorizer.metrics import metrics


class MetricFactory:
    METRICS = {
        'mse': metrics.mse,
        'channelwise_l2': metrics.channelwise_l2,
        'psnr': metrics.psnr,
        'accuracy': metrics.accuracy
    }

    @classmethod
    def get(cls, name):
        if name not in list(cls.METRICS.keys()):
            raise ValueError(f'Invalid value {name} for model name. \
                               Should be on of {lsit(cls.METRICS.keys())}')

        return cls.METRICS[name]


    @classmethod
    def list(cls):
        return list(cls.METRICS.keys())
