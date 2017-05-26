import torch
import numpy as np


def test_diff(filename1, filename2, filename3):
    input1 = torch.load(filename1)
    input2 = torch.load(filename2)
    input3 = torch.load(filename3)

    return np.float32((input1 - input2 - input3).numpy())