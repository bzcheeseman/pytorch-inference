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
  std::vector<pytorch::tensor> tests = test_setup({1, 1, 1},
                                            {2, 2, 2},
                                            {45, 45, 45},
                                            {50, 50, 50},
                                            {1},
                                            {2},
                                            {45},
                                            {50},
                                            {"test_prod1.dat", "test_prod2.dat", "test_prod3.dat"},
                                            "test_prod");

  // tests has {input1, input2, input3, pytorch_output}

  pytorch::Product p(pytorch::k, 3);
  af::timer::start();
  pytorch::tensor prod;
  for (int j = 49; j >= 0; j--){
    prod = p({tests[0], tests[1], tests[2]})[0];
    prod.eval();
  }
  af::sync();
  std::cout << "arrayfire forward took (s): " << af::timer::stop()/50 << "(avg)" << std::endl;

  assert(almost_equal(prod, tests[3]));

}
