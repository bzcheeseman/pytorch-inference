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


#ifndef PYTORCH_INFERENCE_ACTIVATIONS_HPP
#define PYTORCH_INFERENCE_ACTIVATIONS_HPP

// STL
#include <vector>

// ArrayFire
#include <arrayfire.h>

// Project
#include "Layer.hpp"
#include "Activations_Impl.hpp"

namespace pytorch {

  //! @todo: docs
  class Sigmoid : public Layer {
  public:
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::sigmoid(input[0])};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::sigmoid(input[0])};
    }
  };

  //! @todo: docs
  class Tanh : public Layer {
  public:
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::tanh(input[0])};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::tanh(input[0])};
    }
  };

  //! @todo: docs
  class Hardtanh : public Layer {
    const float low, high;
  public:
    Hardtanh(const float &low = -1.f, const float &high = 1.f) : low(low), high(high) {}

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::hardtanh(input[0], low, high)};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::hardtanh(input[0], low, high)};
    }
  };

  //! @todo: docs
  class ReLU : public Layer {
  public:
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::relu(input[0])};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::relu(input[0])};
    }
  };

  //! @todo: docs
  class Softmax : public Layer { // SO SLOW GOOD LORD
  public:
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::softmax(input[0])};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::softmax(input[0])};
    }
  };
} // pytorch

#endif //PYTORCH_INFERENCE_ACTIVATIONS_HPP
