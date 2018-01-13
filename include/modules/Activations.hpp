//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_ACTIVATIONS_HPP
#define PYTORCH_INFERENCE_ACTIVATIONS_HPP

// STL
#include <vector>

// Project
#include "Layer.hpp"
#include "../functional/activations.hpp"
#include "../storage/tensor.hpp"

namespace pytorch {

  //! @todo: docs
  class Sigmoid : public Layer {
  public:
    inline std::vector<tensor> forward(const std::vector<tensor> &input){
      return {functional::sigmoid(input[0])};
    }

    inline std::vector<tensor> operator()(const std::vector<tensor> &input){
      return {functional::sigmoid(input[0])};
    }
  };

  //! @todo: docs
  class Tanh : public Layer {
  public:
    inline std::vector<tensor> forward(const std::vector<tensor> &input){
      return {functional::tanh(input[0])};
    }

    inline std::vector<tensor> operator()(const std::vector<tensor> &input){
      return {functional::tanh(input[0])};
    }
  };

  //! @todo: docs
  class Hardtanh : public Layer {
    const float low, high;
  public:
    explicit Hardtanh(const float &low = -1.f, const float &high = 1.f) : low(low), high(high) {}

    inline std::vector<tensor> forward(const std::vector<tensor> &input){
      return {functional::hardtanh(input[0], low, high)};
    }

    inline std::vector<tensor> operator()(const std::vector<tensor> &input){
      return {functional::hardtanh(input[0], low, high)};
    }
  };

  //! @todo: docs
  class ReLU : public Layer {
  public:
    inline std::vector<tensor> forward(const std::vector<tensor> &input){
      return {functional::relu(input[0])};
    }

    inline std::vector<tensor> operator()(const std::vector<tensor> &input){
      return {functional::relu(input[0])};
    }
  };

  //! @todo: docs
  class Softmax : public Layer { // SO SLOW GOOD LORD
  public:
    inline std::vector<tensor> forward(const std::vector<tensor> &input){
      return {functional::softmax(input[0])};
    }

    inline std::vector<tensor> operator()(const std::vector<tensor> &input){
      return {functional::softmax(input[0])};
    }
  };
} // pytorch

#endif //PYTORCH_INFERENCE_ACTIVATIONS_HPP
