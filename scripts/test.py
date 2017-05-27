import torch
import torch.nn.functional as Funct
from torch.autograd import Variable
import numpy as np


def make_tensor(n, k, h, w=None):
    if w:
        tensor = torch.zeros(n, k, h, w).uniform_(0, .1)
    else:
        tensor = torch.zeros(n, k, h).uniform_(0, .1)
    return np.float32(tensor.numpy())  # this np.float32 cast is crucial


def save_tensor(n, k, h, w, filename):
    tensor = torch.zeros(n, k, h, w).uniform_(0, .1)
    torch.save(tensor, filename)


def load_tensor(filename):
    tensor = torch.load(filename)
    return np.float32(tensor.numpy())


def test_concat(filts_file, dim):
    f = Variable(torch.load(filts_file))
    return np.float32(torch.cat([f, f], dim).data.numpy())


def test_pool(img_file):
    f = Variable(torch.load(img_file))
    pool = torch.nn.AvgPool2d((2, 1), stride=(2, 1))
    out = pool(f)
    return np.float32(out.data.numpy())


def test_conv(filts_file, bias_file, img_file, lw_file, lb_file, gamma_file, beta_file, rm_file, rv_file):
    img = Variable(torch.load(img_file))
    weights = Variable(torch.load(filts_file))
    bias = Variable(torch.load(bias_file)).squeeze()
    lw = Variable(torch.load(lw_file)).squeeze()
    lb = Variable(torch.load(lb_file)).squeeze()
    gamma = torch.nn.Parameter(torch.load(gamma_file)).squeeze()
    beta = torch.nn.Parameter(torch.load(beta_file)).squeeze()
    rm = Variable(torch.load(rm_file)).squeeze()
    rv = Variable(torch.load(rv_file)).squeeze()

    bn = torch.nn.BatchNorm2d(weights.size())
    bn.eval()
    bn.weight = torch.nn.Parameter(gamma.data)
    bn.bias = torch.nn.Parameter(beta.data)
    bn.running_mean = rm.data
    bn.running_var = rv.data

    output = Funct.conv2d(img, weights, bias, padding=(1, 1))
    output = bn(output)
    output = Funct.tanh(output)
    output, idx = Funct.max_pool2d(output, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
    output = Funct.sigmoid(output)
    # output = Funct.max_unpool2d(output, idx, kernel_size=(2, 2), stride=(2, 2))
    # output = Funct.avg_pool2d(output, kernel_size=(2, 2), stride=(2, 2))
    output = Funct.max_pool2d(output, kernel_size=(2, 2), stride=(2, 2))
    output = Funct.hardtanh(output, -0.1, 0.1)
    output = Funct.linear(output.view(output.size(0), -1), lw, lb)
    output = Funct.relu(output)
    output = Funct.softmax(output)
    return np.float32(output.unsqueeze(2).unsqueeze(3).data.numpy())  # don't forget to put back unsqueezes
