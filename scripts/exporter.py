import torch
import torch.nn as nn
import torchvision as tv
import os


def export(module, name):
    filenames = {}
    os.makedirs("../save/"+name, exist_ok=True)
    for key in module.state_dict().keys():
        k = key.split(".")[0]
        filenames[k] = []
        filenames[k].append("../save/"+name+"/"+key+".dat")
        torch.save(module.state_dict()[key], "../save/"+name+"/"+key+".dat")
    return filenames


def emit_module_cpp(module, name, max_repeats):

    statedict = module.state_dict()

    for key in statedict.keys():
        dims = list(statedict[key].size())
        key = key.split(".")
        for i in range(max_repeats+1):
            if "conv%d"%(i+1) in key:
                print(".".join(key), dims)
            elif "bn%d"%(i+1) in key:
                print(".".join(key), [1, dims[0], 1, 1])


def run():
    conv = tv.models.resnet18(True)
    return emit_module_cpp(conv, 'resnet18', 3)

if __name__ == "__main__":
    print(run())
