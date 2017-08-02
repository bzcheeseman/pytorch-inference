//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_POOLING_HPP
#define PYTORCH_INFERENCE_POOLING_HPP

// STL
#include <vector>

// ArrayFire
#include <arrayfire.h>

// Project
#include "Layer.hpp"
#include "Pooling_Impl.hpp"

namespace pytorch {
  /**
   * @class MaxPool2d "include/layers.hpp"
   * @file "include/layers.hpp"
   * @brief Equivalent to MaxPool2d in pytorch (with a caveat)
   *
   * This layer implements the forward pass of pytorch's nn.MaxPool2d module. This holds the pooling indices
   * inside itself - unpooling will be another challenge if it's even possible with this framework. For now,
   * we just store the indices in arrayfire format for future use.
   */
  class MaxPool2d : public Layer {
  private:
    pooling_params_t params;
    af::array indices;
  public:
    /**
     * @brief Constructs the MaxPool2d layer. Requires pooling parameters that are functionally equivalent to
     * the convolutional parameters.
     *
     * @param params Pooling parameters like window, stride, etc.
     */
    MaxPool2d(const pooling_params_t &params) : params(params) {}

    inline af::array get_indices() const {
      return indices;
    }

    /**
     * @brief Implements the forward pass
     *
     * @param input The input array to be pooled
     * @return The pooled array
     */
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::maxpool(params, input[0], indices)};
    }

    /**
     * @brief Implements the forward pass
     *
     * @param input The input array to be pooled
     * @return The pooled array
     */
    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::maxpool(params, input[0], indices)};
    }
  };

  /**
   * @class MaxUnool2d "include/layers.hpp"
   * @file "include/layers.hpp"
   * @brief Equivalent to MaxUnool2d in pytorch
   *
   * This layer implements the forward pass of pytorch's nn.MaxUnool2d module. This holds a pointer to a maxpool
   * layer from which to get the indices for the unpooling process.
   */
  class MaxUnpool2d : public Layer {
  private:
    pooling_params_t params;
    const MaxPool2d *mp_ref;
  public:
    /**
     * @brief Constructs the MaxUnpool2d layer. Requires pooling parameters that are functionally equivalent to
     * the convolutional parameters.
     *
     * @param params Pooling parameters like window, stride, etc.
     */
    MaxUnpool2d(const pooling_params_t &params, const MaxPool2d *mp_ref) : params(params), mp_ref(mp_ref) {}

    /**
     * @brief Implements the forward pass
     *
     * @param input The input array to be unpooled
     * @return The unpooled array
     */
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::unpool(params, input[0], mp_ref->get_indices())};
    }

    /**
     * @brief Implements the forward pass
     *
     * @param input The input array to be unpooled
     * @return The unpooled array
     */
    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::unpool(params, input[0], mp_ref->get_indices())};
    }
  };

  /**
   * @class AvgPool2d "include/layers.hpp"
   * @file "include/layers.hpp"
   * @brief Equivalent to AvgPool2d in pytorch
   *
   * This layer implements the forward pass of pytorch's nn.AvgPool2d module. It doesn't do anything fancy,
   * just takes the mean of the various windows.
   */
  class AvgPool2d : public Layer {
  private:
    pooling_params_t params;
  public:
    /**
     * @brief Constructs the AvgPool2d layer. Requires pooling parameters that are functionally
     * identical to the convolution parameters.
     *
     * @param params Pooling parameters
     */
    AvgPool2d(const pooling_params_t &params) : params(params) {}

    /**
     * @brief Implements the forwards pass
     *
     * @param input The input array to be pooled
     * @return The pooled array
     */
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::avgpool(params, input[0])};
    }

    /**
     * @brief Implements the forwards pass
     *
     * @param input The input array to be pooled
     * @return The pooled array
     */
    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::avgpool(params, input[0])};
    }
  };
} // pytorch

#endif //PYTORCH_INFERENCE_POOLING_HPP
