//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_NORMALIZATION_IMPL_HPP
#define PYTORCH_INFERENCE_NORMALIZATION_IMPL_HPP

// ArrayFire
#include <arrayfire.h>

// Project
#include "../storage/tensor.hpp"

namespace pytorch::functional {
  inline tensor batchnorm2d(const tensor &gamma, // vectors of size C (applied across channels)
                               const tensor &beta,
                               const tensor &running_mean,
                               const tensor &running_variance,
                               const float &epsilon,
                               const tensor &input){

    internal::check_size(gamma.data().dims(2), input.data().dims(2), __func__);
    internal::check_size(beta.data().dims(2), input.data().dims(2), __func__);
    int h_in = input.data().dims(0); int w_in = input.data().dims(1); int batch = input.data().dims(3);

    return (input.data() - af::tile(running_mean.data(), h_in, w_in, 1, batch))
           /(af::sqrt(af::tile(running_variance.data(), h_in, w_in, 1, batch) + epsilon))
           * af::tile(gamma.data(), h_in, w_in, 1, batch) + af::tile(beta.data(), h_in, w_in, 1, batch);
  }
}

#endif //PYTORCH_INFERENCE_NORMALIZATION_IMPL_HPP
