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

namespace pytorch::impl {
  //! @todo: optimize this if possible?
  inline af::array batchnorm2d(const af::array &gamma, // vectors of size C (applied across channels)
                               const af::array &beta,
                               const af::array &running_mean,
                               const af::array &running_variance,
                               const float &epsilon,
                               const af::array &input){

    check_size(gamma.dims(2), input.dims(2), __func__);
    check_size(beta.dims(2), input.dims(2), __func__);
    int h_in = input.dims(0); int w_in = input.dims(1); int batch = input.dims(3);

    af::array out = (input - af::tile(running_mean, h_in, w_in, 1, batch))
                    /(af::sqrt(af::tile(running_variance, h_in, w_in, 1, batch)) + epsilon)
                    * af::tile(gamma, h_in, w_in, 1, batch) + af::tile(beta, h_in, w_in, 1, batch);

    return out;
  }
}

#endif //PYTORCH_INFERENCE_NORMALIZATION_IMPL_HPP
