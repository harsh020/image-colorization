import torch
from torch import nn
from torchvision import models

import config
from colorizer import utils


def mse_loss():
    def mse(y_pred, y_true, *args, **kwargs):
        mse = torch.mean((y_pred-y_true)**2)
        return mse
    return mse


def huber_loss():
    def huber(y_pred, y_true, *args, **kwargs):
        return nn.HuberLoss()(y_pred, y_true)
    return huber


def perceptual_loss(style_weight=0.45, content_weight=0.45, tvr_weight=0.1):
    if config.DEVICE == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = config.DEVICE

    trans = utils.vgg_tensor_transformer()
    vgg = utils.VGG16Extractor().to(device)

    def perceptual_loss_fn(y_pred, y_true, *args, **kwargs):
        y_pred = trans(y_pred)
        y_true = trans(y_true)

        y_true_features = vgg(y_true)
        y_pred_features = vgg(y_pred)

        # l1 loss (colored vs predicted)
        l1_loss = 0.0
        for j in range(4):
            l1_loss += nn.functional.l1_loss(y_pred_features[j], y_true_features[j])

        # style loss (colored vs predicted)
        y_pred_gram = [gram(fmap) for fmap in y_pred_features]
        y_true_gram = [gram(fmap) for fmap in y_true_features]
        style_loss = 0.0
        for j in range(4):
            style_loss += mse(y_pred_gram[j], y_true_gram[j])
        style_loss = style_weight*style_loss
#         aggregate_style_loss += style_loss.data[0]

        # content loss (colored vs predicted)
        true_feat = y_true_features[1]
        pred_feat = y_pred_features[1]
        content_loss = content_weight*mse(pred_feat, true_feat)
#         aggregate_content_loss += content_loss.data[0]

        # total variation regularization
        diff_i = torch.sum(torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))
        tvr_loss = tvr_weight*(diff_i + diff_j)
#         aggregate_tvr_loss += tvr_loss.data[0]

        # total loss
        total_loss = l1_loss + style_loss + content_loss + tvr_loss

        return total_loss

    return perceptual_loss_fn


def contrastive_loss(margin=1.0):

    if config.DEVICE == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = config.DEVICE

    vgg16 = models.vgg16(pretrained=True)
    extractor = nn.Sequential(
        nn.Sequential(*list(vgg16.children())[0][0:31]),
        nn.AvgPool2d(7))

    for params in extractor.parameters():
        params.require_grad = False
    extractor = extractor.to(device)
    extractor.eval()

    trans = utils.vgg_tensor_transformer()

    def contrastive_loss_fn(y_pred, y_true, y_false):
        y_pred = trans(y_pred)
        y_true = trans(y_true)

        new_shape = list(y_false.shape)
        new_shape[1] = 3
        y_false = y_false.expand(new_shape)
        y_false = trans(y_false)

        y_pred_feature = extractor(y_pred)
        y_true_feature = extractor(y_true)
        y_false_feature = extractor(y_false)

        loss = 0.0

        pos = torch.mean(torch.sqrt((y_true_feature-y_pred_feature)**2), dim=-1)
        neg = torch.mean(torch.sqrt((y_false_feature-y_pred_feature)**2), dim=-1)
        ones = torch.ones(pos.shape)
        zeros = torch.zeros(neg.shape)

        y_hat = torch.cat((pos, neg), dim=0).to(device)
        y = torch.cat((ones, zeros), dim=0).to(device)

        loss += torch.mean((0.5*(1-y)*(y_hat**2))+
                           (0.5*y*(nn.functional.relu(margin-y_hat)**2)))
        return loss

    return contrastive_loss_fn
