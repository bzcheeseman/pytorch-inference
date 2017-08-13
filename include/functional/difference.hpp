//
// Created by Aman LaChapelle on 5/26/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_DIFFERENCE_IMPL_HPP
#define PYTORCH_INFERENCE_DIFFERENCE_IMPL_HPP

// STL
#include <algorithm>

// ArrayFire
#include <arrayfire.h>

// Project
#include "../utils.hpp"
#include "../storage/tensor.hpp"

namespace pytorch::functional {

  inline tensor difn(const std::vector<tensor> &inputs,
                        const int &dim){

    int n_tensors = inputs.size();
    af::array out = inputs[0].data();
#pragma omp target device(0) map(out, inputs)
    for (int i = 1; i < n_tensors; i++){
      out -= inputs[i].data();
    }
    return tensor(out);
  }

} // pytorch::functional

#endif //PYTORCH_INFERENCE_DIFFERENCE_IMPL_HPP
