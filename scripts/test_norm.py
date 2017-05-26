import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def test_norm(gamma_file, beta_file, rm_file, rv_file, img_file):
    gamma = torch.nn.Parameter(torch.load(gamma_file)).squeeze()
    beta = torch.nn.Parameter(torch.load(beta_file)).squeeze()
    rm = Variable(torch.load(rm_file)).squeeze()
    rv = Variable(torch.load(rv_file)).squeeze()
    img = Variable(torch.load(img_file))

    bn = torch.nn.BatchNorm2d(3)
    bn.eval()
    bn.weight = torch.nn.Parameter(gamma.data)
    bn.bias = torch.nn.Parameter(beta.data)
    bn.running_mean = rm.data
    bn.running_var = rv.data

    output = bn(img)
    return np.float32(output.data.numpy())
