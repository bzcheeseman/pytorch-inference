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
  std::vector<af::array> tests = test_setup({3}, {4}, {5}, {6},
                                            {3, 3, 3}, {4, 4, 4}, {5, 5, 5}, {6, 6, 6},
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


}
