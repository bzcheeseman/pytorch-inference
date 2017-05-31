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


#ifndef PYTORCH_INFERENCE_LINEAR_IMPL_HPP
#define PYTORCH_INFERENCE_LINEAR_IMPL_HPP

// ArrayFire
#include <arrayfire.h>

namespace pytorch::impl {
  /**
   * @brief Performs the linear transformation y = Wx + b
   *
   * @param weight The weight matrix W
   * @param bias The bias vector b
   * @param input The input x
   * @return The transformed input (a.k.a. output) y
   */
  inline af::array linear(const af::array &input,
                          const af::array &weight,
                          const af::array &bias,
                          const bool &has_bias){
    long batch = input.dims(3);
    long flat = input.dims(0) * input.dims(1) * input.dims(2);

    check_size(weight.dims(1), flat, __func__);

    af::array out = af::matmul(weight, af::moddims(af::transpose(input), flat, batch));
    if (has_bias)
      out += af::tile(bias, 1, batch);

    out = af::reorder(out, 0, 3, 2, 1);

    return out;
  }
}

#endif //PYTORCH_INFERENCE_LINEAR_IMPL_HPP
