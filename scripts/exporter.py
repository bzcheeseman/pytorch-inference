import torch
import torchvision as tv
import os


def export(module, name):
    filenames = {}
    os.makedirs("../save/"+name, exist_ok=True)
    for key in module.state_dict().keys():
        filenames[key] = "../save/"+name+"/"+key+".dat"
        torch.save(module.state_dict()[key], "../save/"+name+"/"+key+".dat")
    return filenames


def run():
    conv = tv.models.resnet18(True)
    return export(conv, 'resnet18')

if __name__ == "__main__":
    print(run())
