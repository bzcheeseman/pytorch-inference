//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_CONCATENATE_HPP
#define PYTORCH_INFERENCE_CONCATENATE_HPP

#include "Layer.hpp"
#include "Concatenate_Impl.hpp"

namespace pytorch {

  class Concat : public Layer {
    int dim;
  public:
    Concat(const int &dim) : dim(dim) {}

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::catn(input, dim)};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::catn(input, dim)};
    }
  };

} // pytorch

#endif //PYTORCH_INFERENCE_CONCATENATE_HPP
