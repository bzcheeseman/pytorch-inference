//
// Created by Aman LaChapelle on 5/26/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_PRODUCT_HPP
#define PYTORCH_INFERENCE_PRODUCT_HPP

#include "Layer.hpp"
#include "Product_Impl.hpp"

namespace pytorch {

  class Product : public Layer {
    int dim;
    int n_tensors;
  public:
    Product(const int &dim, const int &n_tensors) : dim(dim), n_tensors(n_tensors) {}

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::prodn(input, dim)};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::prodn(input, dim)};
    }
  };

} // pytorch

#endif //PYTORCH_INFERENCE_PRODUCT_HPP
