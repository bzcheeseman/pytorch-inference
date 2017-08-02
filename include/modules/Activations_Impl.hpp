//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

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
    return af::max(a, af::constant(0, a.dims()));
  }

  inline af::array softmax(const af::array &a){ // only suitable for vectors(!)
    af::array out (a.dims());
    gfor (af::seq i, a.dims(3)){
      af::array a_vol = a(af::span, af::span, af::span, i);
      out(af::span, af::span, af::span, i) =
              af::exp(a_vol - af::max(a_vol))/af::sum(af::exp(a_vol - af::max(a_vol)));
    }
    return out;
  }
} // pytorch::impl

#endif //PYTORCH_INFERENCE_ACTIVATIONS_IMPL_HPP
