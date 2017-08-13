//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_BRANCH_IMPL_HPP
#define PYTORCH_INFERENCE_BRANCH_IMPL_HPP

// STL
#include <vector>

// ArrayFire
#include <arrayfire.h>

// Project
#include "../utils.hpp"

namespace pytorch::functional {
  inline std::vector<tensor> copy_branch(const tensor &input, const int &copies){
    return std::vector<tensor> (copies, input);
  }
} // pytorch::functional

#endif //PYTORCH_INFERENCE_BRANCH_IMPL_HPP
