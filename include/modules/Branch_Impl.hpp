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


#ifndef PYTORCH_INFERENCE_BRANCH_IMPL_HPP
#define PYTORCH_INFERENCE_BRANCH_IMPL_HPP

// STL
#include <assert.h>
#include <vector>

// ArrayFire
#include <arrayfire.h>

namespace pytorch::impl {
  inline std::vector<af::array> copy_branch(const af::array &input, const int &copies){
    assert(copies <= 4); // only 4 branches supported for now
    std::vector<af::array> out;
    for (int i = 0; i < copies; i++){
      out.push_back(input);
    }
    return out;
  }
} // pytorch::impl

#endif //PYTORCH_INFERENCE_BRANCH_IMPL_HPP
