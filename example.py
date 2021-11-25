# Import packages
# ----------------

# For functional mode
from PIL import Image
from colorizer import colorizer

# For direct classes
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from colorizer import models
from colorizer.train import Trainer
from colorizer.inference import Inferencer
from colorizer.losses.factory import LossFactory
from colorizer.colorize_data import ColorizeData
from colorizer.models.factory import ModelFactory
from colorizer.metrics.factory import MetricFactory



# Example when using functional mode
# ----------------------------------
def train_ex():
    state_dict_path = None # provide path where state dict will be save or provide in config
    model = colorizer.train()


def inference_ex():
    image_path = 'PATH/TO/IMAGE'
    state_dict_path = 'PATH/TO/STATE_DICT' # or provide in config
    colored = colorzer.inference(image_path, state_dict_path)

    colored.show()


# Example when using direct classes
# ----------------------------------

def trainer():
    model = ModelFactory.get('NAME-OF-MODEL')

    train_dataset = ColorizeData('PATH/TO/IMAGES')
    train_dataloader = DataLoader(train_dataset, batch_size='<BATCH_SIZE>', shuffle='<SHUFFLE>')


    valid_dataset = ColorizeData('PATH/TO/IMAGES')
    valid_dataloader = DataLoader(valid_dataset, batch_size='<BATCH_SIZE>', shuffle='<SHUFFLE>')

    loss = LossFactory.get('NAME-OF-LOSS')
    metrics = {
        metric: MetricFactory.get(metric)
        for metric in ['LIST-OF-NAME-OF-METRICS']
    }
    optmizer = # Define your optimizer

    trainer = Trainer(model)
    model.compile(loss, metrics, optimizer)
    model = model.fit(epochs = # Number of epochs,
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader)

    torch.save(model.state_dict(), 'PATH/TO/SAVE/STATE/DICT')


def inferencer():
    model = # Make your model

    state_dict_path = 'PATH/TO/STATE/DICT'
    image = 'PATH/TO/IMAGE'
    save_path = 'PATH/TO/SAVE/COLORED/IMAGE'

    inferencer = Inferencer(model, state_dict_path)
    image = inferencer.colorize(image, save_path)
    return image
