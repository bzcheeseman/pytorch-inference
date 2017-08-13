//
// Created by Aman LaChapelle on 8/13/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#include "../include/layers.hpp"
#include "../include/inference_engine.hpp"



int main() {
  af::array input = af::loadImage("../data/cifar/img_18.jpg", true)/255.f;
  input = af::resize(input, 224, 224);
  af_print(alexnet_forward({pytorch::tensor(input)}).data()); // pytorch output =
  return 0;
}