//
// Created by Aman LaChapelle on 5/26/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "../include/layers.hpp"
#include "../include/inference_engine.hpp"

af::array alexnet_forward(const af::array &input) {
  pytorch::inference_engine engine (1, AF_BACKEND_OPENCL, false);

  pytorch::conv_params_t convparams1 = {11, 11, 4, 4, 2, 2};
  pytorch::Conv2d conv1(convparams1, "../save/alexnet/conv1.weight.dat", {64, 3, 11, 11}, "../save/alexnet/conv1.bias.dat", {1, 64, 1, 1});
  engine.add_layer(&conv1);
  pytorch::ReLU relu1;
  engine.add_layer(&relu1);
  pytorch::pooling_params_t mpparams1 = {3, 3, 2, 2, 0, 0};
  pytorch::MaxPool2d maxpool1(mpparams1);
  engine.add_layer(&maxpool1);
  pytorch::conv_params_t convparams2 = {5, 5, 1, 1, 2, 2};
  pytorch::Conv2d conv2(convparams2, "../save/alexnet/conv2.weight.dat", {192, 64, 5, 5}, "../save/alexnet/conv2.bias.dat", {1, 192, 1, 1});
  engine.add_layer(&conv2);
  pytorch::ReLU relu2;
  engine.add_layer(&relu2);
  pytorch::pooling_params_t mpparams2 = {3, 3, 2, 2, 0, 0};
  pytorch::MaxPool2d maxpool2(mpparams2);
  engine.add_layer(&maxpool2);
  pytorch::conv_params_t convparams3 = {3, 3, 1, 1, 1, 1};
  pytorch::Conv2d conv3(convparams3, "../save/alexnet/conv3.weight.dat", {384, 192, 3, 3}, "../save/alexnet/conv3.bias.dat", {1, 384, 1, 1});
  engine.add_layer(&conv3);
  pytorch::ReLU relu3;
  engine.add_layer(&relu3);
  pytorch::conv_params_t convparams4 = {3, 3, 1, 1, 1, 1};
  pytorch::Conv2d conv4(convparams4, "../save/alexnet/conv4.weight.dat", {256, 384, 3, 3}, "../save/alexnet/conv4.bias.dat", {1, 256, 1, 1});
  engine.add_layer(&conv4);
  pytorch::ReLU relu4;
  engine.add_layer(&relu4);
  pytorch::conv_params_t convparams5 = {3, 3, 1, 1, 1, 1};
  pytorch::Conv2d conv5(convparams5, "../save/alexnet/conv5.weight.dat", {256, 256, 3, 3}, "../save/alexnet/conv5.bias.dat", {1, 256, 1, 1});
  engine.add_layer(&conv5);
  pytorch::ReLU relu5;
  engine.add_layer(&relu5);
  pytorch::pooling_params_t mpparams3 = {3, 3, 2, 2, 0, 0};
  pytorch::MaxPool2d maxpool3(mpparams3);
  engine.add_layer(&maxpool3);
  pytorch::Linear lin1("../save/alexnet/lin1.weight.dat", {1, 1, 4096, 9216}, "../save/alexnet/lin1.bias.dat", {1, 1, 4096, 1});
  engine.add_layer(&lin1);
  pytorch::ReLU relu6;
  engine.add_layer(&relu6);
  pytorch::Linear lin2("../save/alexnet/lin2.weight.dat", {1, 1, 4096, 4096}, "../save/alexnet/lin2.bias.dat", {1, 1, 4096, 1});
  engine.add_layer(&lin2);
  pytorch::ReLU relu7;
  engine.add_layer(&relu7);
  pytorch::Linear lin3("../save/alexnet/lin3.weight.dat", {1, 1, 1000, 4096}, "../save/alexnet/lin3.bias.dat", {1, 1, 1000, 1});
  engine.add_layer(&lin3);
  pytorch::Linear lin4("../save/alexnet/lin4.weight.dat", {1, 1, 10, 1000}, "../save/alexnet/lin4.bias.dat", {1, 1, 10, 1});
  engine.add_layer(&lin4);
  pytorch::Softmax softmax1;
  engine.add_layer(&softmax1);

  af::array output = engine.forward({input});
  output.eval();
  af::sync();
  return output;
}

int main(){ // add softmax
  af::setBackend(AF_BACKEND_OPENCL);
  af::setDevice(1);
  af::array input;
  input = af::loadImage("../data/cifar/img_18.jpg", true)/255.f; // should be a bird (also don't use .png)
  input = af::resize(input, 224, 224);
//  std::cout << input.dims() << std::endl;
  auto output = alexnet_forward(input);
  af_print(output);
}
