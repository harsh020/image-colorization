import config

import numpy as np
from PIL import Image

import torch
from torchvision import transforms


class Inferencer:
    def __init__(self, model, load_path):
        if config.DEVICE == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif config.DEVICE in ['cpu', 'cuda']:
            self.device = config.DEVICE
        else:
            self.device = 'cpu'

        self.model = model

        self.input_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Resize(size=(256, 256))])
        self.build(load_path)

    def build(self, load_path):
        state_dict = torch.load(load_path, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict)

    def colorize(self, image_path, save_path=None):
        if save_path == None:
            save_path = config.SAVE_PATH
        image = Image.open(image_path)
        image = image.convert('L')
        tensor = self.input_transforms(image)
        tensor = torch.unsqueeze(tensor, dim=0)

        with torch.no_grad():
            array = self.model(tensor).detach().cpu().numpy()

        array = np.squeeze(array, axis=0)
        array = np.moveaxis(array, 0, -1)
        color_image = Image.fromarray(np.uint8(array * 255))
        color_image.save(save_path)
        return color_image
