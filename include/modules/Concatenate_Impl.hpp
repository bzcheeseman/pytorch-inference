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

#include <algorithm>

// ArrayFire
#include <arrayfire.h>

// Project
#include "../utils.hpp"

namespace pytorch::impl {

  inline af::array catn(const std::vector<af::array> &inputs,
                        const int &dim){
    check_num_leq(inputs.size(), 10, __func__);

    int n_arrays = inputs.size();

    std::vector<af_array> to_cat (n_arrays);
    af::array out;
    af_array outp = out.get();
    
    std::transform(inputs.begin(), inputs.end(), to_cat.begin(),
                   [](const af::array &a) -> af_array {
                     return a.get();
                   });

    af_join_many(&outp, dim, n_arrays, to_cat.data());

    out = af::array(outp);

    return out;
  }

} // pytorch::impl

#endif //PYTORCH_INFERENCE_CONCATENATE_IMPL_HPP
