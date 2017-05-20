import torch
import torch.nn.functional as Funct
from torch.autograd import Variable
import numpy as np


def make_tensor(n, k, h, w=None):
    if w:
        tensor = torch.rand(n, k, h, w)
    else:
        tensor = torch.rand(n, k, h)
    return np.float32(tensor.numpy())  # this np.float32 cast is crucial


def save_tensor(n, k, h, w, filename):
    tensor = torch.rand(n, k, h, w)
    torch.save(tensor, filename)


def load_tensor(filename):
    tensor = torch.load(filename)
    return np.float32(tensor.numpy())


def test_conv(filts_file, bias_file, img_file, lw_file, lb_file):
    img = Variable(torch.load(img_file))
    weights = Variable(torch.load(filts_file))
    bias = Variable(torch.load(bias_file)).squeeze()
    lw = Variable(torch.load(lw_file)).squeeze()
    lb = Variable(torch.load(lb_file)).squeeze()
    output = Funct.conv2d(img, weights, bias)
    output = Funct.hardtanh(Funct.linear(output.view(output.size(0), -1), lw, lb))
    return np.float32(output.unsqueeze(2).unsqueeze(3).data.numpy())
