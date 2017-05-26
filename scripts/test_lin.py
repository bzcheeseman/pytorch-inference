import torch
import torch.nn.functional as Funct
from torch.autograd import Variable
import numpy as np


def test_lin(w_filename, b_filename, in_filename):
    w = Variable(torch.load(w_filename))
    b = Variable(torch.load(b_filename))
    input = Variable(torch.load(in_filename))

    output = Funct.linear(input.view(input.size(0), -1), w.squeeze(), b.squeeze()).unsqueeze(1).unsqueeze(3)
    return np.float32(output.data.numpy())
