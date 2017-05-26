import torch
import numpy as np


def test_branch(filename):
    input = torch.load(filename)
    return [np.float32(input.numpy()) for i in range(3)]
