//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_CONCATENATE_HPP
#define PYTORCH_INFERENCE_CONCATENATE_HPP

// STL
#include <vector>

// Project
#include "Layer.hpp"
#include "../functional/concatenate.hpp"
#include "../storage/tensor.hpp"

namespace pytorch {

  class Concat : public Layer {
    int dim;
  public:
    explicit Concat(const int &dim) : dim(dim) {}

    inline std::vector<tensor> forward(const std::vector<tensor> &input){
      return {functional::catn(input, dim)};
    }

    inline std::vector<tensor> operator()(const std::vector<tensor> &input){
      return {functional::catn(input, dim)};
    }
  };

} // pytorch

#endif //PYTORCH_INFERENCE_CONCATENATE_HPP
