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
  std::vector<pytorch::tensor> tests = test_setup({3}, {32}, {55}, {55},
                                            {3, 3, 3}, {32, 32, 32}, {55, 55, 55}, {55, 55, 55},
                                            {"test_branch.dat"},
                                            "test_branch");

  // tests now has {input, input_copy1, input_copy2, input_copy3}
  pytorch::Branch b(pytorch::n);

  af::timer::start();
  auto branches = b({tests[0]});
  std::cout << "arrayfire forward took (s): " << af::timer::stop() << std::endl;

  for (int i = 0; i < branches.size(); i++){
    assert(almost_equal(branches[i], tests[i+1]));
  }

  return 0;

}
