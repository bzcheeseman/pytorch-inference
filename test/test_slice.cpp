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
  std::vector<af::array> tests = test_setup({1}, {4}, {80}, {80},
                                            {1, 1}, {2, 2}, {80, 80}, {80, 80},
                                            {"test_slice_img.dat"},
                                            "test_slice");

  // tests now has {img, slice1, slice2}

  pytorch::Slice s(2, pytorch::k);

  af::timer::start();
  auto sliced = s({tests[0]});
  std::cout << "arrayfire forward took (s): " << af::timer::stop() << std::endl;

  assert(almost_equal(sliced[0], tests[1]));
  assert(almost_equal(sliced[1], tests[2]));

}
