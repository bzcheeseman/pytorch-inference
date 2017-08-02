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
  std::vector<af::array> tests = test_setup({1, 1, 3}, {1, 1, 1}, {5, 5, 1}, {600, 1, 600},
                                            {3}, {1}, {5}, {1},
                                            {"test_lin_weight.dat", "test_linear_bias.dat", "test_linear_img.dat"},
                                            "test_lin");

  // tests now has {weight, bias, img, pytorch_out}

  pytorch::Linear l(tests[0], tests[1]);

  af::timer::start();
  af::array lin;
  for (int j = 50-1; j != 0; j--){
    lin = l({tests[2]})[0];
    lin.eval();
  }
  af::sync();
  std::cout << "arrayfire forward took (s): " << af::timer::stop()/50 << "(avg)" << std::endl;

  assert(almost_equal(lin, tests[3]));

}
