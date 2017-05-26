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


#ifndef PYTORCH_INFERENCE_ACTIVATIONS_IMPL_HPP
#define PYTORCH_INFERENCE_ACTIVATIONS_IMPL_HPP

#include <arrayfire.h>

namespace pytorch::impl {
  inline af::array sigmoid(const af::array &a){
    return 1 / (1 + af::exp(-a));
  }

  inline af::array hardtanh(const af::array &a, const float &low=1.f, const float &high=1.f){
    return af::clamp(a, low, high);
  }

  inline af::array tanh(const af::array &a){
    return af::tanh(a);
  }

  inline af::array relu(const af::array &a){
    return af::max(a, af::constant(0, a.dims())); // hopefully this is working like I think it is
  }

  inline af::array softmax(const af::array &a){
    af::array out (a.dims());
    af::array sum (a.dims(3));
    gfor (af::seq i, a.dims(3)){
      af::array a_vol = a(af::span, af::span, af::span, i);
      af::array z = af::exp(a_vol - af::max(a_vol));
      out(af::span, af::span, af::span, i) = z/af::sum(z, 2);
    }
    return out;
  }
} // pytorch::impl

#endif //PYTORCH_INFERENCE_ACTIVATIONS_IMPL_HPP
