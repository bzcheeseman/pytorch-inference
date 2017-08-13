//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_SLICE_IMPL_HPP
#define PYTORCH_INFERENCE_SLICE_IMPL_HPP

// STL
#include <vector>

// ArrayFire
#include <arrayfire.h>

// Project
#include "../storage/tensor.hpp"

namespace pytorch::functional {

  inline std::vector<tensor> split_branch(const tensor &input, const int &slices, const int &dim){
    internal::check_size(input.data().dims(dim)%slices, 0, __func__); // it's gotta be an even split
    std::vector<tensor> out;
    switch (dim) { // each array has a slice that is an equal piece of that dimension. Also af::seq is inclusive.
      case 0 : // change to gfor?
        for (int i = 0; i < input.data().dims(0); i += input.data().dims(0)/slices) {
          out.push_back(tensor(input.data()(af::seq(i, i+input.data().dims(0)/slices-1), af::span, af::span, af::span)));
        }
        break;

      case 1 :
        for (int i = 0; i < input.data().dims(1); i += input.data().dims(1)/slices) {
          out.push_back(tensor(input.data()(af::span, af::seq(i, i+input.data().dims(1)/slices-1), af::span, af::span)));
        }
        break;

      case 2 :
        for (int i = 0; i < input.data().dims(2); i += input.data().dims(2)/slices) {
          out.push_back(tensor(input.data()(af::span, af::span, af::seq(i, i+input.data().dims(2)/slices-1), af::span)));
        }
        break;

      case 3 :
        for (int i = 0; i < input.data().dims(3); i += input.data().dims(3)/slices) {
          out.push_back(tensor(input.data()(af::span, af::span, af::span, af::seq(i, i+input.data().dims(3)/slices-1))));
        }
        break;

      default: out.push_back(tensor(input)); std::cerr << "Dimension not in range, no-op." << std::endl; break;
    }
    return out;
  }

} // pytorch::functional

#endif //PYTORCH_INFERENCE_SLICE_IMPL_HPP
