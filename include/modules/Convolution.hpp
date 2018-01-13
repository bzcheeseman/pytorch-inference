//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_CONVOLUTION_HPP
#define PYTORCH_INFERENCE_CONVOLUTION_HPP

// Python
#include <Python.h>
#include <numpy/ndarraytypes.h>

// STL
#include <assert.h>
#include <string>
#include <vector>

// Project
#include "Layer.hpp"
#include "../functional/convolution.hpp"
#include "../py_object.hpp"
#include "../utils.hpp"

namespace pytorch {
  /**
   * @class Conv2d "include/layers.hpp"
   * @file "include/layers.hpp"
   * @brief Equivalent to Conv2d in pytorch.
   *
   * Implements the forward pass for pytorch's nn.Conv2d module. Note that clearly something needs to happen to
   * get the tensors from python to C++, but I've tried to take care of this through the import constructor.
   */
  class Conv2d : public Layer {
  private:
    tensor filters;
    tensor bias;
    conv_params_t params;
    pycpp::py_object utils;
    bool has_bias = false;

  public:
    /**
     * @brief Constructs a Conv2d object given parameters, filters, and bias tensors.
     *
     * @param params The convolution parameters like filter size, stride, and padding.
     * @param filters The trained filter tensors. For those comfortable with Py_Cpp.
     * @param bias The trained bias tensors. For those comfortable with Py_Cpp.
     */
    explicit Conv2d(const conv_params_t &params,
           const tensor &filters,
           const tensor &bias) : params(params), has_bias(true) {
      int Cout = filters.data().dims(3); int Cin = filters.data().dims(2);
      this->filters = tensor(af::unwrap(filters.data(), params.filter_x, params.filter_y, params.filter_x, params.filter_y, 0, 0));
      this->filters = af::reorder(this->filters.data(), 3, 0, 2, 1);
      this->filters = af::moddims(this->filters.data(), Cout, this->filters.data().dims(1)*Cin); // comment to use nested for functional
      internal::check_size(bias.data().dims(2), Cout, __func__);
      this->bias = bias;
    }

    /**
     * @brief Constructs a Conv2d object given the filenames and sizes of the requisite tensors. Also requires
     * convolution parameters like the other constructor.
     *
     * @param params The convolution parameters like filter size, stride, and padding.
     * @param filters_filename The file where the filters tensor is saved. Will be loaded with numpy.load(filename).
     * @param filt_dims The dimensions of the filter tensor in pytorch convention - (batch, channels, h, w)
     * @param bias_filename The file where the bias tensor is saved. Will be loaded with numpy.load(filename).
     * @param bias_dims The dimensions of the bias tensor in pytorch convention - (batch, channels, h, w)
     * @param python_home Where the utility scripts are - holds the loading script necessary to load up the tensors.
     */
    explicit Conv2d(const conv_params_t &params,
           const std::string &filters_filename = "",
           const std::vector<int> &filt_dims = {},
           const std::string &bias_filename = "",
           const std::vector<int> &bias_dims = {},
           const std::string &python_home = "../scripts") : params(params),
                                                            utils("utils", python_home) {

      if (!filters_filename.empty()){
        this->add_filters(filters_filename, filt_dims);
      }

      if (!bias_filename.empty()){
        this->has_bias = true;
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
     * TODO: add filter/bias saving on destroy, corresponding constructor to remake layer
     */
    virtual ~Conv2d() = default;

    /**
     * @brief Read in filters from a file given here if it wasn't passed to the constructor. Overwrites
     * current contents of this->filters.
     *
     * @param filters_filename The file where the filters tensor is saved. Will be loaded with numpy.load(filename).
     * @param filt_dims The dimensions of the filter tensor in pytorch convention - (batch, channels, h, w)
     */
    inline void add_filters(const std::string &filters_filename,
                            const std::vector<int> &filt_dims){
      assert(filt_dims.size() > 0);
      _object *filts = utils("load_array", {pycpp::to_python(filters_filename)}, {});
      assert(filts);
      filters = internal::from_numpy(reinterpret_cast<PyArrayObject *>(filts), filt_dims.size(), filt_dims);
      int Cout = filters.data().dims(3); int Cin = filters.data().dims(2);
      filters = af::unwrap(filters.data(), params.filter_x, params.filter_y, params.filter_x, params.filter_y, 0, 0);
      filters = af::reorder(filters.data(), 3, 0, 2, 1);
      filters = af::moddims(filters.data(), Cout, filters.data().dims(1)*Cin); // comment to use nested for functional
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
      internal::check_size(bias.data().dims(2), filters.data().dims(0), __func__);
      this->has_bias = true;
    }

    /**
     * @brief Forward function, takes data and performs the Conv2d operation using the already-initialized
     * filters and bias tensors
     *
     * @param input Input data size (h_in, w_in, Cin, batch)
     * @return Convolved data size (h_out, w_out, Cout, batch)
     */
    inline std::vector<tensor> forward(const std::vector<tensor> &input){
      return {functional::conv2d(params, input[0], filters, bias, this->has_bias)};
    }

    /**
     * @brief Forward function, takes data and performs the Conv2d operation using the already-initialized
     * filters and bias tensors
     *
     * @param input Input data size (h_in, w_in, Cin, batch)
     * @return Convolved data size (h_out, w_out, Cout, batch)
     */
    inline std::vector<tensor> operator()(const std::vector<tensor> &input){
      return {functional::conv2d(params, input[0], filters, bias, this->has_bias)};
    }

  };
} // pytorch

#endif //PYTORCH_INFERENCE_CONVOLUTION_HPP
