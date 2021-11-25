import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import config
from colorizer import models
from colorizer.train import Trainer
from colorizer.inference import Inferencer
from colorizer.losses.factory import LossFactory
from colorizer.colorize_data import ColorizeData
from colorizer.models.factory import ModelFactory
from colorizer.metrics.factory import MetricFactory


OPTIMIZERS = {
    'adam': optim.Adam,
    'sgd': optim.SGD
}


def train(state_dict_path=None):
    model = ModelFactory.get(config.MODEL)(pretrained=config.PRETRINED_BACKBONE,
                                           freeze=config.FREEZE)

    train_dataset = ColorizeData(config.TRAIN_DATASET_PATH)

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE)

    if config.VALID_DATASET_PATH:
        valid_dataset = ColorizeData(config.VALID_DATASET_PATH)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    else:
        valid_dataloader = None

    loss = LossFactory.get(config.LOSS)
    metrics = {
        metric: MetricFactory.get(metric)
        for metric in config.METRICS
    }
    optimizer = OPTIMIZERS[config.OPTIMIZER](model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.STEP_SIZE,
                                          gamma=config.GAMMA)

    trainer = Trainer(model)
    trainer.compile(loss, metrics, optimizer, scheduler)
    model = trainer.fit(epochs=config.EPOCHS,
                        train_dataloader=train_dataloader,
                        val_dataloader=valid_dataloader)

    if state_dict_path == None:
        state_dict_path = config.STATE_DICT_PATH

    torch.save(model.state_dict(), state_dict_path)
    return model


def inference(image_path, state_dict_path=None, save_path=None):
    model = ModelFactory.get(config.MODEL)(pretrained=config.PRETRINED_BACKBONE,
                                           freeze=config.FREEZE)


    if state_dict_path == None:
        state_dict_path = config.STATE_DICT_PATH

    inferencer = Inferencer(model, state_dict_path)
    image = inferencer.colorize(image_path, save_path)
    return image
