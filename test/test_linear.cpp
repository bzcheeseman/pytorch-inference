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
  std::vector<af::array> tests = test_setup({1, 1, 3}, {1, 1, 1}, {5, 5, 1}, {600, 1, 600},
                                            {3}, {1}, {5}, {1},
                                            {"test_lin_weight.dat", "test_linear_bias.dat", "test_linear_img.dat"},
                                            "test_lin");

  // tests now has {weight, bias, img, pytorch_out}

  pytorch::Linear l(tests[0], tests[1]);

  af::timer::start();
  auto lin = l({tests[2]})[0];
  std::cout << "arrayfire forward took (s): " << af::timer::stop() << std::endl;

  assert(almost_equal(lin, tests[3]));

}
