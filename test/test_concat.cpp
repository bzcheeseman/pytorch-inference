//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#include "../include/layers.hpp"

#include "utils.hpp"

int main(){
  std::vector<af::array> tests = test_setup({1, 1, 1},
                                            {2, 3, 4},
                                            {45, 45, 45},
                                            {50, 50, 50},
                                            {1},
                                            {9},
                                            {45},
                                            {50},
                                            {"test_concat1.dat", "test_concat2.dat", "test_concat3.dat"},
                                            "test_concat");

  // tests has {input1, input2, input3, pytorch_output}

  pytorch::Concat c(pytorch::k);
  af::timer::start();
  auto catted = c({tests[0], tests[1], tests[2]})[0];
  std::cout << "arrayfire forward took (s): " << af::timer::stop() << std::endl;

  assert(almost_equal(catted, tests[3]));

  return 0;

}

