//
// Created by Aman LaChapelle on 5/19/17.
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


#ifndef PYTORCH_INFERENCE_LAYERS_HPP
#define PYTORCH_INFERENCE_LAYERS_HPP

#include <arrayfire.h>

#include "ops.hpp"
#include "utils.hpp"
#include "py_object.hpp"

namespace pytorch {

  /**
   * @brief Abstract base class for all layers.
   */
  class Layer { // forward and operator() are no-ops for the base class
  public:
    inline virtual af::array forward(const af::array &input) = 0;

    inline virtual af::array operator()(const af::array &input) = 0;
  };

  /*  Conv2d  */
  class Conv2d : public Layer {
  private:
    af::array filters;
    af::array bias;
    conv_params_t params;
    pycpp::py_object utils;

  public:
    Conv2d(const conv_params_t &params,
           const af::array &filters,
           const af::array &bias) : params(params), filters(filters), bias(bias) { }

    Conv2d(const conv_params_t &params,
           const std::string &filters_filename,
           const std::vector<int> &filt_dims,
           const std::string &bias_filename,
           const std::vector<int> &bias_dims,
           const std::string &python_home = "../scripts") : params(params), utils("utils", python_home) {

      PyObject *filts = utils("load_tensor", {pycpp::to_python(filters_filename)});
      assert(filts);
      PyObject *bs = utils("load_tensor", {pycpp::to_python(bias_filename)});
      assert(bs);

      filters = from_numpy(reinterpret_cast<PyArrayObject *>(filts), filt_dims.size(), filt_dims);
      bias = from_numpy(reinterpret_cast<PyArrayObject *>(bs), bias_dims.size(), bias_dims);

    }

    Conv2d(const Conv2d &other){
      filters = other.filters;
      bias = other.bias;
      params = other.params;
    }

    virtual ~Conv2d() {}

    inline af::array forward(const af::array &input){
      return conv2d(params, filters, bias, input);
    }

    inline af::array operator()(const af::array &input){
      return conv2d(params, filters, bias, input);
    }

  };

  /*  Linear  */
  class Linear : public Layer {
  private:
    af::array weights;
    af::array bias;
    pycpp::py_object utils;

  public:
    Linear(const af::array &weights,
           const af::array &bias) : weights(weights), bias(bias) { }

    Linear(const std::string &weights_filename,
           const std::vector<int> &weights_dims,
           const std::string &bias_filename,
           const std::vector<int> &bias_dims,
           const std::string &python_home = "../scripts") : utils("utils", python_home) {

      PyObject *ws = utils("load_tensor", {pycpp::to_python(weights_filename)});
      assert(ws);
      PyObject *bs = utils("load_tensor", {pycpp::to_python(bias_filename)});
      assert(bs);

      weights = from_numpy(reinterpret_cast<PyArrayObject *>(ws), weights_dims.size(), weights_dims);
      bias = from_numpy(reinterpret_cast<PyArrayObject *>(bs), bias_dims.size(), bias_dims);

    }

    Linear(const Linear &other){
      weights = other.weights;
      bias = other.bias;
    }

    virtual ~Linear() {}

    inline af::array forward(const af::array &input){
      return linear(weights, bias, input);
    }

    inline af::array operator()(const af::array &input){
      return linear(weights, bias, input);
    }

  };

  // TODO: MaxPool
  // TODO: BatchNorm (af::transform? Or just apply regular batchnorm?)

  /* Sigmoid */
  class Sigmoid : public Layer {
  public:
    inline af::array forward(const af::array &input){
      return pytorch::sigmoid(input);
    }

    inline af::array operator()(const af::array &input){
      return pytorch::sigmoid(input);
    }
  };

  /* Tanh */
  class Tanh : public Layer {
  public:
    inline af::array forward(const af::array &input){
      return pytorch::tanh(input);
    }

    inline af::array operator()(const af::array &input){
      return pytorch::tanh(input);
    }
  };

  /* Hardtanh */
  class Hardtanh : public Layer {
    const float low, high;
  public:
    Hardtanh(const float &low = 1.f, const float &high = 1.f) : low(low), high(high) {}

    inline af::array forward(const af::array &input){
      return pytorch::hardtanh(input, low, high);
    }

    inline af::array operator()(const af::array &input){
      return pytorch::hardtanh(input, low, high);
    }
  };

  /* ReLU */
  class ReLU : public Layer {
  public:
    inline af::array forward(const af::array &input){
      return pytorch::relu(input);
    }

    inline af::array operator()(const af::array &input){
      return pytorch::relu(input);
    }
  };

}  // pytorch



#endif //PYTORCH_INFERENCE_LAYERS_HPP
