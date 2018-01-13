//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_LAYER_HPP_HPP
#define PYTORCH_INFERENCE_LAYER_HPP_HPP

// STL
#include <vector>

// ArrayFire
#include <arrayfire.h>

// Project
#include "../storage/tensor.hpp"

namespace pytorch {
  /**
   * @class Layer
   * @file "include/layers.hpp"
   * @brief Abstract base class for all layers.
   *
   * We store pointers to this class in the inference engine which allows us to use a std::vector to store
   * them but still have multiple classes represented. The only requirement is that they implement the
   * forward and operator() methods.
   */
  class Layer {
  public:
    /**
     * @brief Forward function for this layer
     * @param input The input to this layer
     * @return The output of this layer
     */
    inline virtual std::vector<tensor> forward(const std::vector<tensor> &input) = 0;

    /**
     * @brief Forward function for this layer
     * @param input The input to this layer
     * @return The output of this layer
     */
    inline virtual std::vector<tensor> operator()(const std::vector<tensor> &input) = 0;
  };

  /**
   * @class Skip "include/layers.hpp"
   * @file "include/layers.hpp"
   * @brief Performs a no-op - makes it so that we can create branching networks.
   */
  class Skip : public Layer {
  public:
    /**
     * @brief No-op forward
     * @param input Input tensor(s)
     * @return Input tensor(s)
     */
    inline std::vector<tensor> forward(const std::vector<tensor> &input) override {
      return input;
    }

    /**
     * @brief No-op forward
     * @param input Input tensor(s)
     * @return Input tensor(s)
     */
    inline std::vector<tensor> operator()(const std::vector<tensor> &input) override {
      return input;
    }
  };
} // pytorch

#endif //PYTORCH_INFERENCE_LAYER_HPP_HPP
