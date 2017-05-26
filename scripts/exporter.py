import torch
from torch.autograd import Variable
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


def emit_module_cpp(var, params):

    param_map = {id(v): k for k, v in params.items()}

    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                print(size_to_str(var.size()))
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map[id(u)], size_to_str(u.size()))
                print(node_name)
            else:
                print(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        # dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    # dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var)


def run():
    inputs = torch.randn(1,3,224,224)

    resnet18 = tv.models.resnet18(True)
    y = resnet18(Variable(inputs))
    emit_module_cpp(y, resnet18.state_dict())

if __name__ == "__main__":
    print(run())
