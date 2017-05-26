import torch
import torch.nn.functional as Funct
from torch.autograd import Variable
import numpy as np


def test_act(filename):
    input = Variable(torch.load(filename))
    sigmoid = Funct.sigmoid(input)
    tanh = Funct.tanh(input)
    hardtanh = Funct.hardtanh(input, -2.5, 2.5)
    relu = Funct.relu(input)
    softmax = Funct.softmax(input)

    return [
        np.float32(sigmoid.data.numpy()),
        np.float32(tanh.data.numpy()),
        np.float32(hardtanh.data.numpy()),
        np.float32(relu.data.numpy()),
        np.float32(softmax.data.numpy())
    ]
