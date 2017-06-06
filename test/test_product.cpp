//
// Created by Aman LaChapelle on 5/26/17.
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
  af::array prod;
  for (int j = 49; j >= 0; j--){
    prod = p({tests[0], tests[1], tests[2]})[0];
    prod.eval();
  }
  af::sync();
  std::cout << "arrayfire forward took (s): " << af::timer::stop()/50 << "(avg)" << std::endl;

  assert(almost_equal(prod, tests[3]));

}
