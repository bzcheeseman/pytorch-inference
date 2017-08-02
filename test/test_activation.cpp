//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#include "../include/layers.hpp"

#include "utils.hpp"

int main() {
  std::vector<af::array> tests = test_setup({30}, {1}, {500}, {1},
                                            {30, 30, 30, 30, 30},
                                            {1, 1, 1, 1, 1},
                                            {500, 500, 500, 500, 500},
                                            {1, 1, 1, 1, 1},
                                            {"test_activation.dat"},
                                            "test_act");

  // Tests has {input_tensor, sigmoid_out, tanh_out, hardtanh_out, relu_out, softmax_out}
  pytorch::Sigmoid s; pytorch::Tanh t; pytorch::Hardtanh ht (-2.5f, 2.5f); pytorch::ReLU r; pytorch::Softmax so;

  af::timer::start();
  auto sigmoid_out = s({tests[0]})[0];
  auto tanh_out = t({tests[0]})[0];
  auto hardtanh_out = ht({tests[0]})[0];
  auto relu_out = r({tests[0]})[0];
  auto softmax_out = so({tests[0]})[0];
  std::cout << "arrayfire forward took (s): " << af::timer::stop() << std::endl;

  assert(almost_equal(sigmoid_out, tests[1]));
  assert(almost_equal(tanh_out, tests[2]));
  assert(almost_equal(hardtanh_out, tests[3]));
  assert(almost_equal(relu_out, tests[4]));
  assert(almost_equal(softmax_out, tests[5]));

  return 0;

}
