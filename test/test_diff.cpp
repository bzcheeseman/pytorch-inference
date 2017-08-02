//
// Created by Aman LaChapelle on 5/26/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#include "../include/layers.hpp"

#include "utils.hpp"

int main(){
  std::vector<af::array> tests = test_setup({1, 1, 1},
                                            {2, 2, 2},
                                            {45, 45, 45},
                                            {50, 50, 50},
                                            {1},
                                            {2},
                                            {45},
                                            {50},
                                            {"test_diff1.dat", "test_diff2.dat", "test_diff3.dat"},
                                            "test_diff");

  // tests has {input1, input2, input3, pytorch_output}

  pytorch::Difference d(pytorch::k, 3);
  af::timer::start();
  auto diff = d({tests[0], tests[1], tests[2]})[0];
  std::cout << "arrayfire forward took (s): " << af::timer::stop() << std::endl;

  assert(almost_equal(diff, tests[3]));

}

