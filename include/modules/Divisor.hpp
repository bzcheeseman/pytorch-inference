//
// Created by Aman LaChapelle on 5/26/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_DIVISOR_HPP
#define PYTORCH_INFERENCE_DIVISOR_HPP

#include "Layer.hpp"
#include "../functional/divisor.hpp"
#include "../storage/tensor.hpp"

namespace pytorch {

  class Divisor : public Layer {
    int dim;
    int n_tensors;
  public:
    Divisor(const int &dim, const int &n_tensors) : dim(dim), n_tensors(n_tensors) {}

    inline std::vector<tensor> forward(const std::vector<tensor> &input){
      return {functional::divn(input, dim)};
    }

    inline std::vector<tensor> operator()(const std::vector<tensor> &input){
      return {functional::divn(input, dim)};
    }
  };

} // pytorch

#endif //PYTORCH_INFERENCE_DIVISOR_HPP
