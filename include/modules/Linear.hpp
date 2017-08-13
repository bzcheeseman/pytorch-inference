//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_LINEAR_HPP
#define PYTORCH_INFERENCE_LINEAR_HPP

// Python
#include <Python.h>
#include <numpy/ndarraytypes.h>

// STL
#include <assert.h>
#include <string>
#include <vector>

// Project
#include "Layer.hpp"
#include "../functional/linear.hpp"
#include "../py_object.hpp"
#include "../utils.hpp"

namespace pytorch {
  /**
   * @class Linear "include/layers.hpp"
   * @file "include/layers.hpp"
   * @brief Equivalent to Linear in pytorch
   *
   * Implements the forward pass for pytorch's nn.Linear module. Note that clearly something needs to happen to
   * get the tensors from python to C++, but I've tried to take care of this through the import constructor.
   */
  class Linear : public Layer {
  private:
    tensor weights;
    tensor bias;
    pycpp::py_object utils;
    bool has_bias = false;

  public:
    /**
     * @brief Constructs a Linear object given weights, and bias tensors.

     * @param weights The trained weight tensors. For those comfortable with Py_Cpp.
     * @param bias The trained bias tensors. For those comfortable with Py_Cpp. Can be initialized to zero.
     */
    Linear(const tensor &weights,
           const tensor &bias) : weights(weights), bias(bias), has_bias(true) {
      internal::check_size(bias.data().dims(0), weights.data().dims(0), __func__);
    }

    /**
     * @brief Constructs a Linear object given the filenames and sizes of the requisite tensors.
     *
     * @param weights_filename The file where the weights tensor is saved. Will be loaded with numpy.load(filename).
     * @param weights_dims The dimensions of the weights tensor in pytorch convention - (batch, channels, h, w)
     * @param bias_filename The file where the bias tensor is saved. Will be loaded with numpy.load(filename).
     * @param bias_dims The dimensions of the bias tensor in pytorch convention - (batch, channels, h, w)
     * @param python_home Where the utility scripts are - holds the loading script necessary to load up the tensors.
     */
    Linear(const std::string &weights_filename = "",
           const std::vector<int> &weights_dims = {},
           const std::string &bias_filename = "",
           const std::vector<int> &bias_dims = {},
           const std::string &python_home = "../scripts") : utils("utils", python_home) {

      if (!weights_filename.empty()){
        this->add_weights(weights_filename, weights_dims);
      }
      if (!bias_filename.empty()){
        this->has_bias = true;
        this->add_bias(bias_filename, bias_dims);
      }
    }

    /**
     * @brief Copy constructor, constructs another Linear object that is an exact copy of the argument.
     *
     * @param other Another Linear object to copy.
     */
    Linear(const Linear &other){
      weights = other.weights;
      bias = other.bias;
    }

    /**
     * @brief Destructor - for now trivial, may need to take on some functionality.
     * TODO: add filter/bias saving on destroy, corresponding constructor to remake layer
     */
    virtual ~Linear() {}

    /**
     * @brief Read in weights from a file given here if it wasn't passed to the constructor. Overwrites
     * current contents of this->weights.
     *
     * @param weights_filename The file where the weights tensor is saved. Will be loaded with numpy.load(filename).
     * @param weights_dims The dimensions of the weights tensor in pytorch convention - (batch, channels, h, w)
     */
    inline void add_weights(const std::string &weights_filename,
                            const std::vector<int> &weights_dims){
      assert(weights_dims.size() > 0);
      _object *ws = utils("load_array", {pycpp::to_python(weights_filename)}, {});
      assert(ws);
      weights = internal::from_numpy(reinterpret_cast<PyArrayObject *>(ws), weights_dims.size(), weights_dims);
    }

    /**
     * @brief Read in bias from a file given here if it wasn't passed to the constructor. Overwrites
     * current contents of this->bias.
     *
     * @param bias_filename The file where the bias tensor is saved. Will be loaded with numpy.load(filename).
     * @param bias_dims The dimensions of the bias tensor in pytorch convention - (batch, channels, h, w)
     */
    inline void add_bias(const std::string &bias_filename,
                         const std::vector<int> &bias_dims){
      assert(bias_dims.size() > 0);
      _object *bs = utils("load_array", {pycpp::to_python(bias_filename)}, {});
      assert(bs);
      bias = internal::from_numpy(reinterpret_cast<PyArrayObject *>(bs), bias_dims.size(), bias_dims);
      internal::check_size(bias.data().dims(0), weights.data().dims(0), __func__);
      this->has_bias = true;
    }

    /**
     * @brief Forward function, takes data and performs the Linear operation using the already-initialized
     * weights and bias tensors
     *
     * @param input Input data size (dims_in, 1, 1, batch)
     * @return Transformed data size (dims_out, 1, batch)
     */
    inline std::vector<tensor> forward(const std::vector<tensor> &input){
      return {functional::linear(input[0], weights, bias, this->has_bias)};
    }

    /**
     * @brief Forward function, takes data and performs the Linear operation using the already-initialized
     * weights and bias tensors
     *
     * @param input Input data size (dims_in, 1, 1, batch)
     * @return Transformed data size (dims_out, 1, batch)
     */
    inline std::vector<tensor> operator()(const std::vector<tensor> &input){
      return {functional::linear(input[0], weights, bias, this->has_bias)};
    }

  };
} // pytorch

#endif //PYTORCH_INFERENCE_LINEAR_HPP
