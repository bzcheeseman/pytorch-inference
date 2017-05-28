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


#ifndef PYTORCH_INFERENCE_CONCATENATE_IMPL_HPP
#define PYTORCH_INFERENCE_CONCATENATE_IMPL_HPP

// ArrayFire
#include <arrayfire.h>

// Project
#include "../utils.hpp"

namespace pytorch::impl {

  inline af::array catn(const std::vector<af::array> &inputs,
                        const int &dim){
    check_num_leq(inputs.size(), 10, __func__);

    int n_arrays = inputs.size();

    af_array *to_cat = new af_array [n_arrays];
    af_array *catted = new af_array;

#pragma clang loop unroll(enable)
    for (int i = 0; i < n_arrays; i++){
      to_cat[i] = inputs[i].get();
    }

    af_join_many(catted, dim, n_arrays, to_cat);

    af::array out (*catted);

    delete[](to_cat);
    delete(catted);

    return out;
  }

} // pytorch::impl

#endif //PYTORCH_INFERENCE_CONCATENATE_IMPL_HPP
