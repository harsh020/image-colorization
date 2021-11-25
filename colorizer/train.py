import sys

import torch
from torch import nn, optim
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader

import config


class Trainer:
    def __init__(self, model):
        if config.DEVICE == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif config.DEVICE in ['cpu', 'cuda']:
            self.device = config.DEVICE
        else:
            raise ValueError(f'Invalid value {config.DEVICE} for `device`. \
                               Allowed values cpu, cuda')

        self.model = model.to(self.device)
        self._delim = '\n'

    def compile(self, loss, metrics, optimizer, scheduler):
        self.criterion = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.scheduler = scheduler

    def train_step(self, x, y):
        self.optimizer.zero_grad()

        predicted = self.model(x)
        loss = self.criterion(predicted, y)
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        metrics = {}
        for name, metric in self.metrics.items():
            metrics[name] = metric(predicted, y)

        del x, y
        return loss, metrics

    def val_step(self, x, y):
        with torch.no_grad():
            predicted = self.model(x)
            loss = self.criterion(predicted, y)

            metrics = {}
            for name, metric in self.metrics.items():
                metrics[name] = metric(predicted, y)

        del x, y
        return loss, metrics

    def train(self, train_dataloader):
        train_loss = 0.0
        train_metrics = {
            metric: 0.0
            for metric in self.metrics.keys()
        }

        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            loss, metrics = self.train_step(batch_x, batch_y)

            train_loss += loss.item()
            for metric, val in metrics.items():
                train_metrics[metric] += val.item()

        train_loss /= len(train_dataloader)
        template = f'loss: {round(train_loss, 4)}, '
        for metric, val in metrics.items():
                train_metrics[metric] /= len(train_dataloader)
                template += f'{metric}: {round(train_metrics[metric], 4)}, '
        print(template, end=self._delim, flush=True)

        del loss, metrics
        return train_loss, train_metrics

    def validate(self, val_dataloader):
        val_loss = 0.0
        val_metrics = {
            metric: 0.0
            for metric in self.metrics.keys()
        }

        for batch_x, batch_y in val_dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            loss, metrics = self.val_step(batch_x, batch_y)

            val_loss += loss.item()
            for metric, val in metrics.items():
                val_metrics[metric] += val.item()

        val_loss /= len(val_dataloader)
        template = f'val_loss: {round(val_loss, 4)}, '
        for metric, val in metrics.items():
                val_metrics[metric] /= len(val_dataloader)
                template += f'val_{metric}: {round(val_metrics[metric], 4)}, '
        print(template, end=self._delim, flush=True)

        del loss, metrics
        return val_loss, val_metrics

    def fit(self, train_dataloader, val_dataloader=None, epochs=1):
        self._delim = ', '
        print(f'{self.device} detected!')

        for epoch in range(epochs):
            print(f'Epochs: {(epoch+1):3d}/{epochs}', end=self._delim, flush=True)

            train = self.train(train_dataloader)

            if val_dataloader:
                val = self.validate(val_dataloader)
            print()

        return self.model
