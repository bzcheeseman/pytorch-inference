//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_BRANCH_HPP
#define PYTORCH_INFERENCE_BRANCH_HPP

// Project
#include "Layer.hpp"
#include "Branch_Impl.hpp"

namespace pytorch {
  //! @todo: docs
  class Branch : public Layer {
  private:
    int copies;
  public:
    Branch(const int &copies) : copies(copies){}

    inline int get_copies() const {
      return copies;
    }

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return impl::copy_branch(input[0], copies);
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return impl::copy_branch(input[0], copies);
    }
  };
} // pytorch

#endif //PYTORCH_INFERENCE_BRANCH_HPP
