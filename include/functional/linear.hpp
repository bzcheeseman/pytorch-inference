//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_LINEAR_IMPL_HPP
#define PYTORCH_INFERENCE_LINEAR_IMPL_HPP

// ArrayFire
#include <arrayfire.h>

// Project
#include "../storage/tensor.hpp"

namespace pytorch::functional {
  /**
   * @brief Performs the linear transformation y = Wx + b
   *
   * @param weight The weight matrix W
   * @param bias The bias vector b
   * @param input The input x
   * @return The transformed input (a.k.a. output) y
   */
  inline tensor linear(const tensor &input,
                          const tensor &weight,
                          const tensor &bias,
                          const bool &has_bias){
    long batch = input.data().dims(3);
    long flat = input.data().dims(0) * input.data().dims(1) * input.data().dims(2);

    internal::check_size(weight.data().dims(1), flat, __func__);

    af::array out = af::matmul(weight.data(), af::moddims(af::transpose(input.data()), flat, batch));
    if (has_bias)
      out += af::tile(bias.data(), 1, batch);

    out = af::reorder(out, 0, 3, 2, 1);

    return tensor(out);
  }
}

#endif //PYTORCH_INFERENCE_LINEAR_IMPL_HPP
