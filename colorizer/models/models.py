import torch.nn as nn
import torchvision.models as models


def block(in_filters, out_filters, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_filters, out_filters, kernel_size=kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_filters),
        nn.ReLU()
    )


class ColorNet18(nn.Module):
  def __init__(self, input_size=256, pretrained=True, freeze=True):
    super(ColorNet18, self).__init__()


    # ResNet - First layer accepts grayscale images,
    # and we take only the first few layers of ResNet for this task
    resnet = models.resnet18(pretrained=pretrained)
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])
    RESNET_FEATURE_SIZE = 128

    if freeze:
        for params in self.midlevel_resnet.parameters():
            params.require_grad = False

    ## Upsampling Network
    self.upsample = nn.Sequential(
      nn.Conv2d(RESNET_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):
    midlevel_features = self.midlevel_resnet(input)
    output = self.upsample(midlevel_features)
    return output


class ConstantNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ConstantNet, self).__init__()

        self.block_1 = block(1, 32)
        self.block_2 = block(32, 64)
        self.block_3 = block(64, 128)
        self.block_4 = block(128, 64)
        self.block_5 = block(64, 32)
        self.block_6 = block(32, 3)

    def forward(self, x):
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)
        x_3 = self.block_3(x_2)
        x_4 = x_2 + self.block_4(x_3)
        x_5 = self.block_5(x_4)
        x_6 = self.block_6(x_5)

        del x_1, x_2, x_3, x_4, x_5

        return x_6
