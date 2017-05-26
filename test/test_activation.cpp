//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "../include/layers.hpp"

#include "utils.hpp"

int main() {
  std::vector<af::array> tests = test_setup({3}, {4}, {5}, {6},
                                            {3, 3, 3, 3, 3}, {4, 4, 4, 4, 4}, {5, 5, 5, 5, 5}, {6, 6, 6, 6, 6},
                                            {"test_activation.dat"},
                                            "test_act");

  // Tests has {input_tensor, sigmoid_out, tanh_out, hardtanh_out, relu_out, softmax_out}
  pytorch::Sigmoid s; pytorch::Tanh t; pytorch::Hardtanh ht (-2.5f, 2.5f); pytorch::ReLU r; pytorch::Softmax so;

  af::timer::start();
  auto sigmoid_out = s({tests[0]})[0];
  auto tanh_out = t({tests[0]})[0];
  auto hardtanh_out = ht({tests[0]})[0];
  auto relu_out = r({tests[0]})[0];
  auto softmax_out = so({tests[0]})[0];
  std::cout << "arrayfire forward took (s): " << af::timer::stop() << std::endl;

  assert(almost_equal(sigmoid_out, tests[1]));
  assert(almost_equal(tanh_out, tests[2]));
  assert(almost_equal(hardtanh_out, tests[3]));
  assert(almost_equal(relu_out, tests[4]));
  assert(almost_equal(softmax_out, tests[5]));

  return 0;

}
