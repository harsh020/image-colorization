import torch
from torch import nn
from torchvision import models, transforms


class VGG16Extractor(nn.Module):
    def __init__(self):
        super(VGG16Extractor, self).__init__()

        vgg = models.vgg16(pretrained=True)
        features = vgg.features

        self.relu_1_2 = nn.Sequential()
        self.relu_2_2 = nn.Sequential()
        self.relu_3_3 = nn.Sequential()
        self.relu_4_3 = nn.Sequential()

        for x in range(4):
            self.relu_1_2.add_module(str(x), features[x])

        for x in range(4, 9):
            self.relu_2_2.add_module(str(x), features[x])

        for x in range(9, 16):
            self.relu_3_3.add_module(str(x), features[x])

        for x in range(16, 23):
            self.relu_4_3.add_module(str(x), features[x])

        for params in self.parameters():
            params.requires_grad = False

    def forward(self, input):
        h_relu_1_2 = self.relu_1_2(input)
        h_relu_2_2 = self.relu_2_2(h_relu_1_2)
        h_relu_3_3 = self.relu_3_3(h_relu_2_2)
        h_relu_4_3 = self.relu_4_3(h_relu_3_3)

        return h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3


def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

def vgg_tensor_transformer():
    transformer = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225]),
                                      transforms.Resize(size=(224, 224))])
    return transformer
