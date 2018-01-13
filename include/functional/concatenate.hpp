//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_CONCATENATE_IMPL_HPP
#define PYTORCH_INFERENCE_CONCATENATE_IMPL_HPP

// STL
#include <algorithm>
#include <numeric>

// ArrayFire
#include <arrayfire.h>

// Project
#include "../utils.hpp"

namespace pytorch::functional {

  inline tensor catn(const std::vector<tensor> &inputs,
                        const int &dim){

    af::array out;

    out = std::accumulate(inputs.begin(), inputs.end(), out,
                          [dim](const af::array &init, const tensor &to_cat) -> af::array {
                            return af::join(dim, init, to_cat.data());
                          });

    return tensor(out);
  }

} // pytorch::functional

#endif //PYTORCH_INFERENCE_CONCATENATE_IMPL_HPP
