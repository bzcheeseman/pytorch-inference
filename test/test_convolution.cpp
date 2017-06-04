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
  std::vector<af::array> tests = test_setup({64, 1, 5}, {3, 64, 3}, {3, 1, 226}, {3, 1, 226},
                                            {5}, {64}, {224}, {224},
                                            {"test_conv_filter.dat", "test_conv_bias.dat", "test_conv_img.dat"},
                                            "test_conv");

  // tests now has {filters, bias, img, pytorch_out}

  pytorch::conv_params_t params = {3, 3, 1, 1, 0, 0};

  pytorch::Conv2d c(params, tests[0], tests[1]);

  af::timer::start();
  auto conv = c({tests[2]})[0];
  std::cout << "arrayfire forward took (s): " << af::timer::stop() << std::endl;

  assert(almost_equal(conv, tests[3]));

}
