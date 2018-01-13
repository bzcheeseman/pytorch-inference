//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_BRANCH_HPP
#define PYTORCH_INFERENCE_BRANCH_HPP

// STL
#include <vector>

// Project
#include "Layer.hpp"
#include "../functional/branch.hpp"
#include "../storage/tensor.hpp"

namespace pytorch {
  //! @todo: docs
  class Branch : public Layer {
  private:
    int copies;
  public:
    explicit Branch(const int &copies) : copies(copies){}

    inline int get_copies() const {
      return copies;
    }

    inline std::vector<tensor> forward(const std::vector<tensor> &input){
      return functional::copy_branch(input[0], copies);
    }

    inline std::vector<tensor> operator()(const std::vector<tensor> &input){
      return functional::copy_branch(input[0], copies);
    }
  };
} // pytorch

#endif //PYTORCH_INFERENCE_BRANCH_HPP
