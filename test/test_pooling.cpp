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
  af::setBackend(AF_BACKEND_CPU);
  std::vector<af::array> tests = test_setup({5}, {32}, {64}, {64},
                                            {5, 5, 5}, {32, 32, 32}, {33, 64, 33}, {64, 64, 64},
                                            {"test_pool_img.dat"},
                                            "test_pool");

  // tests now has {img, maxpool, max_unpool, avgpool}
  
  pytorch::pooling_params_t params = {2, 1, 2, 1, 1, 0};

  pytorch::MaxPool2d mp (params); pytorch::MaxUnpool2d ump (params, &mp); pytorch::AvgPool2d ap (params);

  af::timer::start();
  auto maxpool = mp({tests[0]})[0];
  auto maxunpool = ump({maxpool})[0];
  auto avgpool = ap({tests[0]})[0];
  std::cout << "arrayfire forward took (s): " << af::timer::stop() << std::endl;

  assert(almost_equal(maxpool, tests[1]));
  assert(almost_equal(maxunpool, tests[2]));
  assert(almost_equal(avgpool, tests[3]));

}
