import glob

import numpy as np
from PIL import Image

from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader

class ColorizeData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filenames = glob.glob(f'{root_dir}/*.jpg')

        self.input_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Resize(size=(256, 256)),
                                                    transforms.Grayscale(num_output_channels=1)])

        self.target_transforms = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize(size=(256, 256))])


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert('RGB')
        image = np.array(image)

        gray = self.input_transforms(image) / 255.
        color = self.target_transforms(image) / 255.

        return gray, color
