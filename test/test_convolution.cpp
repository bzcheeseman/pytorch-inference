//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#include "../include/layers.hpp"
#include "../include/storage/tensor.hpp"

#include "utils.hpp"

int main(){
  std::vector<pytorch::tensor> tests = test_setup({64, 1, 5}, {3, 64, 3}, {7, 1, 226}, {7, 1, 226},
                                            {5}, {64}, {110}, {110},
                                            {"test_conv_filter.dat", "test_conv_bias.dat", "test_conv_img.dat"},
                                            "test_conv");

  // tests now has {filters, bias, img, pytorch_out}

  pytorch::conv_params_t params = {7, 7, 2, 2, 0, 0};

  pytorch::Conv2d c(params, tests[0], tests[1]);

  af::timer::start();
  pytorch::tensor conv;
  for (int j = 50-1; j != 0; j--){
    conv = c({tests[2]})[0].data();
    conv.eval();
  }
  af::sync();
  std::cout << "arrayfire forward took (s): " << af::timer::stop()/50 << "(avg)" << std::endl;

  assert(almost_equal(conv, tests[3]));

}
