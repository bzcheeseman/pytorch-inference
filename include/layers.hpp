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
   * @brief Convenience enum to use whenever you need to specify a dimension (like in Concat)
   */
  enum dims {
    n = 3,
    k = 2,
    h = 0,
    w = 1
  };

  /**
   * @class Layer
   * @file include/layers.hpp
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
    inline virtual std::vector<af::array> forward(const std::vector<af::array> &input) = 0;

    /**
     * @brief Forward function for this layer
     * @param input The input to this layer
     * @return The output of this layer
     */
    inline virtual std::vector<af::array> operator()(const std::vector<af::array> &input) = 0;
  };

  /**
   * @class Skip (TESTED)
   * @file include/layers.hpp
   * @brief Performs a no-op - makes it so that we can create branching networks.
   */
  class Skip : public Layer {
  public:
    /**
     * @brief No-op forward
     * @param input Input tensor(s)
     * @return Input tensor(s)
     */
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return input;
    }

    /**
     * @brief No-op forward
     * @param input Input tensor(s)
     * @return Input tensor(s)
     */
    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return input;
    }
  };

  /**
   * @class Conv2d (TESTED)
   * @file include/layers.hpp
   * @brief Equivalent to Conv2d in pytorch.
   *
   * Implements the forward pass for pytorch's nn.Conv2d module. Note that clearly something needs to happen to
   * get the tensors from python to C++, but I've tried to take care of this through the import constructor.
   */
  class Conv2d : public Layer {
  private:
    af::array filters;
    af::array bias;
    conv_params_t params;
    pycpp::py_object utils;

  public:
    /**
     * @brief Constructs a Conv2d object given parameters, filters, and bias tensors.
     *
     * @param params The convolution parameters like filter size, stride, and padding.
     * @param filters The trained filter tensors. For those comfortable with Py_Cpp.
     * @param bias The trained bias tensors. For those comfortable with Py_Cpp.
     */
    Conv2d(const conv_params_t &params,
           const af::array &filters,
           const af::array &bias) : params(params), filters(filters), bias(bias) { }

    /**
     * @brief Constructs a Conv2d object given the filenames and sizes of the requisite tensors. Also requires
     * convolution parameters like the other constructor.
     *
     * @param params The convolution parameters like filter size, stride, and padding.
     * @param filters_filename The file where the filters tensor is saved. Will be loaded with torch.load(filename).
     * @param filt_dims The dimensions of the filter tensor in pytorch convention - (batch, channels, h, w)
     * @param bias_filename The file where the bias tensor is saved. Will be loaded with torch.load(filename).
     * @param bias_dims The dimensions of the bias tensor in pytorch convention - (batch, channels, h, w)
     * @param python_home Where the utility scripts are - holds the loading script necessary to load up the tensors.
     */
    Conv2d(const conv_params_t &params,
           const std::string &filters_filename = "",
           const std::vector<int> &filt_dims = {},
           const std::string &bias_filename = "",
           const std::vector<int> &bias_dims = {},
           const std::string &python_home = "../scripts") : params(params), utils("utils", python_home) {

      if (!filters_filename.empty()){
        this->add_filters(filters_filename, filt_dims);
      }
      if (!bias_filename.empty()){
        this->add_bias(bias_filename, bias_dims);
      }

    }

    /**
     * @brief Copy constructor, constructs a Conv2d object that is exactly a copy of the argument.
     *
     * @param other Another Conv2d object.
     */
    Conv2d(const Conv2d &other){
      filters = other.filters;
      bias = other.bias;
      params = other.params;
    }

    /**
     * @brief Destructor - for now trivial, may need to take on some functionality.
     */
    virtual ~Conv2d() {}

    /**
     * @brief Read in filters from a file given here if it wasn't passed to the constructor. Overwrites
     * current contents of this->filters.
     *
     * @param filters_filename The file where the filters tensor is saved. Will be loaded with torch.load(filename).
     * @param filt_dims The dimensions of the filter tensor in pytorch convention - (batch, channels, h, w)
     */
    inline void add_filters(const::std::string &filters_filename,
                            const std::vector<int> &filt_dims){
      assert(filt_dims.size() > 0);
      PyObject *filts = utils("load_tensor", {pycpp::to_python(filters_filename)});
      assert(filts);
      filters = from_numpy(reinterpret_cast<PyArrayObject *>(filts), filt_dims.size(), filt_dims);
    }

    /**
     * @brief Read in bias from a file given here if it wasn't passed to the constructor. Overwrites
     * current contents of this->bias.
     *
     * @param bias_filename The file where the bias tensor is saved. Will be loaded with torch.load(filename).
     * @param bias_dims The dimensions of the bias tensor in pytorch convention - (batch, channels, h, w)
     */
    inline void add_bias(const std::string &bias_filename,
                         const std::vector<int> &bias_dims){
      assert(bias_dims.size() > 0);
      PyObject *bs = utils("load_tensor", {pycpp::to_python(bias_filename)});
      assert(bs);
      bias = from_numpy(reinterpret_cast<PyArrayObject *>(bs), bias_dims.size(), bias_dims);
    }

    /**
     * @brief Forward function, takes data and performs the Conv2d operation using the already-initialized
     * filters and bias tensors
     *
     * @param input Input data size (h_in, w_in, Cin, batch)
     * @return Convolved data size (h_out, w_out, Cout, batch)
     */
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {conv2d(params, filters, bias, input[0])};
    }

    /**
     * @brief Forward function, takes data and performs the Conv2d operation using the already-initialized
     * filters and bias tensors
     *
     * @param input Input data size (h_in, w_in, Cin, batch)
     * @return Convolved data size (h_out, w_out, Cout, batch)
     */
    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {conv2d(params, filters, bias, input[0])};
    }

  };

  // TODO: Grouped convolution(?)

  /*  Linear - tested  */
  class Linear : public Layer {
  private:
    af::array weights;
    af::array bias;
    pycpp::py_object utils;

  public:
    Linear(const af::array &weights,
           const af::array &bias) : weights(weights), bias(bias) { }

    Linear(const std::string &weights_filename = "",
           const std::vector<int> &weights_dims = {},
           const std::string &bias_filename = "",
           const std::vector<int> &bias_dims = {},
           const std::string &python_home = "../scripts") : utils("utils", python_home) {

      if (!weights_filename.empty()){
        this->add_weights(weights_filename, weights_dims);
      }
      if (!bias_filename.empty()){
        this->add_bias(bias_filename, bias_dims);
      }
    }

    Linear(const Linear &other){
      weights = other.weights;
      bias = other.bias;
    }

    virtual ~Linear() {}

    inline void add_weights(const std::string &weights_filename,
                            const std::vector<int> &weights_dims){
      assert(weights_dims.size() > 0);
      PyObject *ws = utils("load_tensor", {pycpp::to_python(weights_filename)});
      assert(ws);
      weights = from_numpy(reinterpret_cast<PyArrayObject *>(ws), weights_dims.size(), weights_dims);
    }

    inline void add_bias(const std::string &bias_filename,
                         const std::vector<int> &bias_dims){
      assert(bias_dims.size() > 0);
      PyObject *bs = utils("load_tensor", {pycpp::to_python(bias_filename)});
      assert(bs);
      bias = from_numpy(reinterpret_cast<PyArrayObject *>(bs), bias_dims.size(), bias_dims);
    }

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {linear(weights, bias, input[0])};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {linear(weights, bias, input[0])};
    }

  };

  // TODO: MaxPool
  // TODO: BatchNorm (af::transform? Or just apply regular batchnorm?)

  /* Branch - tested */
  class Branch : public Layer {
  private:
    int copies;
  public:
    Branch(const int &copies) : copies(copies){}

    inline int get_copies(){
      return copies;
    }

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return pytorch::copy_branch(input[0], copies);
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return pytorch::copy_branch(input[0], copies);
    }
  };

  /* Concat2 - tested */
  class Concat2 : public Layer {
    int dim;
  public:
    Concat2 (const int &dim) : dim(dim) {}

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {pytorch::cat2(input[0], input[1], dim)};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {pytorch::cat2(input[0], input[1], dim)};
    }
  };

  /* Concat3 - tested */
  class Concat3 : public Layer {
    int dim;
  public:
    Concat3 (const int &dim) : dim(dim) {}

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {pytorch::cat3(input[0], input[1], input[2], dim)};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {pytorch::cat3(input[0], input[1], input[2], dim)};
    }
  };

  /* Concat4 - higher is not supported by arrayfire - tested */
  class Concat4 : public Layer {
    int dim;
  public:
    Concat4 (const int &dim) : dim(dim) {}

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {pytorch::cat4(input[0], input[1], input[2], input[3], dim)};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {pytorch::cat4(input[0], input[1], input[2], input[3], dim)};
    }
  };

  /* Sigmoid - tested */
  class Sigmoid : public Layer {
  public:
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {pytorch::sigmoid(input[0])};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {pytorch::sigmoid(input[0])};
    }
  };

  /* Tanh - tested */
  class Tanh : public Layer {
  public:
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {pytorch::tanh(input[0])};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {pytorch::tanh(input[0])};
    }
  };

  /* Hardtanh - tested */
  class Hardtanh : public Layer {
    const float low, high;
  public:
    Hardtanh(const float &low = 1.f, const float &high = 1.f) : low(low), high(high) {}

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {pytorch::hardtanh(input[0], low, high)};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {pytorch::hardtanh(input[0], low, high)};
    }
  };

  /* ReLU - tested */
  class ReLU : public Layer {
  public:
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {pytorch::relu(input[0])};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {pytorch::relu(input[0])};
    }
  };

} // pytorch



#endif //PYTORCH_INFERENCE_LAYERS_HPP
