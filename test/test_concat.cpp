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

int main(){
  std::vector<af::array> tests = test_setup({1, 1, 1},
                                            {2, 3, 4},
                                            {45, 45, 45},
                                            {50, 50, 50},
                                            {1},
                                            {9},
                                            {45},
                                            {50},
                                            {"test_concat1.dat", "test_concat2.dat", "test_concat3.dat"},
                                            "test_concat");

  // tests has {input1, input2, input3, pytorch_output}

  pytorch::Concat c(pytorch::k, 3);
  af::timer::start();
  auto catted = c({tests[0], tests[1], tests[2]})[0];
  std::cout << "arrayfire forward took (s): " << af::timer::stop() << std::endl;

  assert(almost_equal(catted, tests[3]));

}

