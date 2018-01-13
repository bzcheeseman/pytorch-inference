//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_SLICE_HPP
#define PYTORCH_INFERENCE_SLICE_HPP

#include "Layer.hpp"
#include "../functional/slice.hpp"

namespace pytorch {

  //! @todo: docs
  class Slice : public Layer {
  private:
    int dim;
    int slices;
  public:
    Slice(const int &dim, const int &slices) : dim(dim), slices(slices) {}

    inline int get_dim() const {
      return dim;
    }

    inline std::vector<tensor> forward(const std::vector<tensor> &input){
      return functional::split_branch(input[0], slices, dim);
    }

    inline std::vector<tensor> operator()(const std::vector<tensor> &input){
      return functional::split_branch(input[0], slices, dim);
    }

  };

}

#endif //PYTORCH_INFERENCE_SLICE_HPP
