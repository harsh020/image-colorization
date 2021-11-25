import torch

def channelwise_l2(y_pred, y_true):
    norm = torch.norm(y_pred-y_true, dim=(2, 3))**2
    sm = torch.sum(norm, dim=1)
    l2 = torch.mean(sm, dim=0)
    return l2

def mse(y_pred, y_true):
    mse = torch.mean((y_pred-y_true)**2)
    return mse

def psnr(y_pred, y_true, threshold=1.0):
    def _clip(data, threshold):
        data[data < 0.0] = 0.0
        data[data > threshold] = threshold
        return data

    y_pred = _clip(y_pred, threshold)
    y_true = _clip(y_true, threshold)
    mse = torch.mean((y_pred - y_true) ** 2)
    return 10 * torch.log10(1 / mse)

def accuracy(y_pred, y_true):
    mask = (y_pred == y_true).type(torch.DoubleTensor)
    return torch.mean(mask)
