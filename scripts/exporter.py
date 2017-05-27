import torch
import torch.nn as nn
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


def emit_module_cpp(module, name):

    conv_counter = 1
    bn_counter = 1
    relu_counter = 1
    tanh_counter = 1
    sigmoid_counter = 1
    softmax_counter = 1
    mp_counter = 1
    ap_counter = 1
    linear_counter = 1

    output_string_overall = "#include \"../include/layers.hpp\"\n#include \"../include/inference_engine.hpp\"\n\n"
    output_string_overall += "af::array %s_forward(const af::array &input) {\n" % name
    output_string_overall += "pytorch::inference_engine engine;\n\n"

    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):

            out_string = "pytorch::conv_params_t convparams%d = {%d, %d, %d, %d, %d, %d};\n" % \
                         (conv_counter, mod.kernel_size[0], mod.kernel_size[1], mod.stride[0], mod.stride[1],
                          mod.padding[0], mod.padding[1])

            out_string += "pytorch::Conv2d conv%d(convparams%d" % (conv_counter, conv_counter)

            for key in mod.state_dict().keys():
                k = key.split(".")[0]
                save_name = "../save/"+name+"/"+"conv%d." % conv_counter + key + ".dat"
                tensor = mod.state_dict()[key]
                if len(tensor.size()) != 4:
                    tensor = tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                torch.save(tensor, save_name)
                dims = list(tensor.size())

                out_string += ", \"%s\", {%d, %d, %d, %d}" % (save_name, dims[0], dims[1], dims[2], dims[3])

            out_string += ");\nengine.add_layer(&conv%d);\n"%conv_counter
            conv_counter += 1
            output_string_overall += out_string
            # print(out_string)

        elif isinstance(mod, nn.BatchNorm2d):

            out_string = "pytorch::BatchNorm2d bn%d(" % bn_counter

            for key in mod.state_dict().keys():
                k = key.split(".")[0]
                save_name = "../save/"+name+"/"+"bn%d." % bn_counter + key + ".dat"
                tensor = mod.state_dict()[key].unsqueeze(0).unsqueeze(2).unsqueeze(3)
                torch.save(tensor, save_name)
                dims = list(tensor.size())

                out_string += "\"%s\", {%d, %d, %d, %d}, " % (save_name, dims[0], dims[1], dims[2], dims[3])

            out_string = out_string[:-2]
            out_string += ");\nengine.add_layer(&bn%d);\n"%bn_counter
            bn_counter += 1
            output_string_overall += out_string
            # print(out_string)

        elif isinstance(mod, nn.ReLU):

            out_string = "pytorch::ReLU relu%d;" % relu_counter
            out_string += "\nengine.add_layer(&relu%d);\n"%relu_counter

            relu_counter += 1
            output_string_overall += out_string
            # print(out_string)

        elif isinstance(mod, nn.Tanh):

            out_string = "pytorch::Tanh tanh%d;" % tanh_counter
            out_string += "\nengine.add_layer(&tanh%d);\n"%tanh_counter

            tanh_counter += 1
            output_string_overall += out_string

        elif isinstance(mod, nn.Sigmoid):

            out_string = "pytorch::Sigmoid sigmoid%d;" % sigmoid_counter
            out_string += "\nengine.add_layer(&sigmoid%d);\n"%sigmoid_counter

            sigmoid_counter += 1
            output_string_overall += out_string

        elif isinstance(mod, nn.Softmax):

            out_string = "pytorch::Softmax softmax%d;" % softmax_counter
            out_string += "\nengine.add_layer(&softmax%d);\n"%softmax_counter

            softmax_counter += 1
            output_string_overall += out_string

        elif isinstance(mod, nn.MaxPool2d):

            out_string = "pytorch::pooling_params_t mpparams%d = {%d, %d, %d, %d, %d, %d};\n" % \
                         (mp_counter, mod.kernel_size, mod.kernel_size, mod.stride, mod.stride,
                          mod.padding, mod.padding)

            out_string += "pytorch::MaxPool2d maxpool%d(mpparams%d" % (mp_counter, mp_counter)

            for key in mod.state_dict().keys():
                k = key.split(".")[0]
                save_name = "../save/"+name+"/"+"maxpool%d." % mp_counter + key + ".dat"
                torch.save(mod.state_dict()[key], save_name)
                dims = list(mod.state_dict()[key].size())

                out_string += ", \"%s\", {%d, %d, %d, %d}" % (save_name, dims[0], dims[1], dims[2], dims[3])

            out_string += ");\nengine.add_layer(&maxpool%d);\n"%mp_counter
            mp_counter += 1
            output_string_overall += out_string
            # print(out_string)

        elif isinstance(mod, nn.AvgPool2d):
            out_string = "pytorch::pooling_params_t apparams%d = {%d, %d, %d, %d, %d, %d};\n" % \
                         (ap_counter, mod.kernel_size, mod.kernel_size, mod.stride, mod.stride,
                          mod.padding, mod.padding)

            out_string += "pytorch::AvgPool2d avgpool%d(apparams%d" % (ap_counter, ap_counter)

            for key in mod.state_dict().keys():
                k = key.split(".")[0]
                save_name = "../save/"+name+"/"+"avgpool%d." % ap_counter + key + ".dat"
                torch.save(mod.state_dict()[key], save_name)
                dims = list(mod.state_dict()[key].size())

                out_string += ", \"%s\", {%d, %d, %d, %d}" % (save_name, dims[0], dims[1], dims[2], dims[3])

            out_string += ");\nengine.add_layer(&avgpool%d);\n"%ap_counter
            ap_counter += 1
            output_string_overall += out_string
            # print(out_string)

        elif isinstance(mod, nn.Linear):

            out_string = "pytorch::Linear lin%d(" % linear_counter

            for key in mod.state_dict().keys():
                k = key.split(".")[0]
                save_name = "../save/"+name+"/"+"lin%d." % linear_counter + key + ".dat"
                tensor = mod.state_dict()[key].unsqueeze(0).unsqueeze(1)
                if len(tensor.size()) != 4:
                    tensor = tensor.unsqueeze(3)
                torch.save(tensor, save_name)
                dims = list(tensor.size())

                out_string += "\"%s\", {%d, %d, %d, %d}, " % (save_name, dims[0], dims[1], dims[2], dims[3])

            out_string = out_string[:-2]
            out_string += ");\nengine.add_layer(&lin%d);\n"%linear_counter
            linear_counter += 1
            output_string_overall += out_string
            # print(out_string)

    output_string_overall += "\naf::array output = engine.forward({input});\noutput.eval();\naf::sync();" \
                             "\nreturn output;\n}"
    return output_string_overall


def run():
    net = tv.models.alexnet(True)
    print(emit_module_cpp(net, "alexnet"))

if __name__ == "__main__":
    run()
