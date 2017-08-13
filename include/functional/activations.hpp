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

#include "../storage/tensor.hpp"

namespace pytorch::functional {
  inline tensor sigmoid(const tensor &a){ // returns a tensor object
    return tensor(1 / (1 + af::exp(-a.data())));
  }

  inline void sigmoid(tensor &a){ // inplace sigmoid
    a = 1 / (1 + af::exp(-a.data()));
  }


  inline tensor hardtanh(const tensor &a, const float &low=1.f, const float &high=1.f){
    return tensor(af::clamp(a.data(), low, high));
  }

  inline void hardtanh(tensor &a, const float &low=1.f, const float &high=1.f){
    a = af::clamp(a.data(), low, high);
  }

  inline tensor tanh(const tensor &a){
    return tensor(af::tanh(a.data()));
  }

  inline void tanh(tensor &a){
    a = af::tanh(a.data());
  }

  inline tensor relu(const tensor &a){
    return tensor(af::max(a.data(), af::constant(0, a.data().dims())));
  }

  inline void relu(tensor &a){
    a = af::max(a.data(), af::constant(0, a.data().dims()));
  }

  inline tensor softmax(const tensor &a){
    af::array out (a.data().dims());
    gfor (af::seq i, a.data().dims(3)){
      af::array a_vol = a.data()(af::span, af::span, af::span, i);
      out(af::span, af::span, af::span, i) =
              af::exp(a_vol - af::max(a_vol))/af::sum(af::exp(a_vol - af::max(a_vol)));
    }
    return tensor(out);
  }

  inline void softmax(tensor &a){
    af::array out (a.data().dims());
    gfor (af::seq i, a.data().dims(3)){
      af::array a_vol = a.data()(af::span, af::span, af::span, i);
      out(af::span, af::span, af::span, i) =
              af::exp(a_vol - af::max(a_vol))/af::sum(af::exp(a_vol - af::max(a_vol)));
    }
    a = out;
  }

} // pytorch::functional

#endif //PYTORCH_INFERENCE_ACTIVATIONS_IMPL_HPP
