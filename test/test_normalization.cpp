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
  std::vector<af::array> tests = test_setup({1, 1, 1, 1, 2}, {32, 32, 32, 32, 32}, {1, 1, 1, 1, 70}, {1, 1, 1, 1, 70},
                                            {2}, {32}, {70}, {70},
                                            {"test_norm_g.dat", "test_norm_b.dat", "test_norm_rm.dat",
                                             "test_norm_rv.dat", "test_norm_img.dat"},
                                            "test_norm");

  // tests now has {gamma, beta, rm, rv, img, pytorch_out}

  pytorch::BatchNorm2d bn(tests[0], tests[1], tests[2], tests[3]);

  af::timer::start();
  auto lin = bn({tests[4]})[0];
  std::cout << "arrayfire forward took (s): " << af::timer::stop() << std::endl;

  assert(almost_equal(lin, tests[5]));

}
