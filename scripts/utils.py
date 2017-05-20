import torch
import numpy as np


def load_tensor(filename):
    tensor = torch.load(filename)
    return np.float32(tensor.numpy())
