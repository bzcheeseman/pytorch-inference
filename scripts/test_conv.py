import torch
from torch.autograd import Variable
import torch.nn.functional as Funct
import numpy as np


def test_conv(filts_file, bias_file, img_file):
    filts = Variable(torch.load(filts_file))
    bias = Variable(torch.load(bias_file))
    img = Variable(torch.load(img_file))

    output = Funct.conv2d(img, filts, bias.squeeze(), stride=(2, 2))
    return np.float32(output.data.numpy())
