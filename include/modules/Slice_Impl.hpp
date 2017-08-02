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

namespace pytorch::impl {

  inline std::vector<af::array> split_branch(const af::array &input, const int &slices, const int &dim){
    check_size(input.dims(dim)%slices, 0, __func__); // it's gotta be an even split
    check_num_leq(slices, 10, __func__); // we can only concatenate 10 items so don't split more than that
    std::vector<af::array> out;
    switch (dim) { // each array has a slice that is an equal piece of that dimension. Also af::seq is inclusive.
      case 0 : // change to gfor?
        for (int i = 0; i < input.dims(0); i += input.dims(0)/slices) {
          out.push_back(input(af::seq(i, i+input.dims(0)/slices-1), af::span, af::span, af::span));
        }
        break;

      case 1 :
        for (int i = 0; i < input.dims(1); i += input.dims(1)/slices) {
          out.push_back(input(af::span, af::seq(i, i+input.dims(1)/slices-1), af::span, af::span));
        }
        break;

      case 2 :
        for (int i = 0; i < input.dims(2); i += input.dims(2)/slices) {
          out.push_back(input(af::span, af::span, af::seq(i, i+input.dims(2)/slices-1), af::span));
        }
        break;

      case 3 :
        for (int i = 0; i < input.dims(3); i += input.dims(3)/slices) {
          out.push_back(input(af::span, af::span, af::span, af::seq(i, i+input.dims(3)/slices-1)));
        }
        break;

      default: out.push_back(input); std::cerr << "Dimension not in range, no-op." << std::endl; break;
    }
    return out;
  }

} // pytorch::impl

#endif //PYTORCH_INFERENCE_SLICE_IMPL_HPP
