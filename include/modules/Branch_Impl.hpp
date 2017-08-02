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

namespace pytorch::impl {
  inline std::vector<af::array> copy_branch(const af::array &input, const int &copies){
    check_num_leq(copies, 10, __func__);
    std::vector<af::array> out (copies, input);
    return out;
  }
} // pytorch::impl

#endif //PYTORCH_INFERENCE_BRANCH_IMPL_HPP
