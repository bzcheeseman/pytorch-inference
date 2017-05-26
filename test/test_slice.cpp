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
  std::vector<af::array> tests = test_setup({1}, {4}, {8}, {8},
                                            {1, 1}, {2, 2}, {8, 8}, {8, 8},
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
