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
                                            {"test_div1.dat", "test_div2.dat", "test_div3.dat"},
                                            "test_div");

  // tests has {input1, input2, input3, pytorch_output}

  pytorch::Divisor d(pytorch::k, 3);
  af::timer::start();
  auto div = d({tests[0], tests[1], tests[2]})[0];
  std::cout << "arrayfire forward took (s): " << af::timer::stop() << std::endl;

  assert(almost_equal(div, tests[3]));

}

