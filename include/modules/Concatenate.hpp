//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef PYTORCH_INFERENCE_CONCATENATE_HPP
#define PYTORCH_INFERENCE_CONCATENATE_HPP

#include "Layer.hpp"
#include "Concatenate_Impl.hpp"

namespace pytorch {
  //! @todo: docs
  class Concat2 : public Layer {
    int dim;
  public:
    Concat2 (const int &dim) : dim(dim) {}

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::cat2(input[0], input[1], dim)};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::cat2(input[0], input[1], dim)};
    }
  };

//! @todo: docs
  class Concat3 : public Layer {
    int dim;
  public:
    Concat3 (const int &dim) : dim(dim) {}

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::cat3(input[0], input[1], input[2], dim)};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::cat3(input[0], input[1], input[2], dim)};
    }
  };

//! @todo: docs
  class Concat4 : public Layer {
    int dim;
  public:
    Concat4 (const int &dim) : dim(dim) {}

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::cat4(input[0], input[1], input[2], input[3], dim)};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::cat4(input[0], input[1], input[2], input[3], dim)};
    }
  };
} // pytorch

#endif //PYTORCH_INFERENCE_CONCATENATE_HPP
