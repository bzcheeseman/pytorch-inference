import torch
import numpy as np


def test_slice(img_filename):
    img = torch.load(img_filename)

    sliced = torch.unbind(img, dim=1)
    s0 = torch.stack(sliced[0:2], dim=1)
    s1 = torch.stack(sliced[2:], dim=1)
    return [
        np.float32(s0.numpy()),
        np.float32(s1.numpy())
    ]
