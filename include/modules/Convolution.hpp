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
#include "Convolution_Impl.hpp"
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
    af::array filters;
    af::array bias;
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
    Conv2d(const conv_params_t &params,
           const af::array &filters,
           const af::array &bias) : params(params), has_bias(true) {
      int Cout = filters.dims(3); int Cin = filters.dims(2);
      this->filters = af::unwrap(filters, params.filter_x, params.filter_y, params.filter_x, params.filter_y, 0, 0);
      this->filters = af::reorder(this->filters, 3, 0, 2, 1);
      this->filters = af::moddims(this->filters, Cout, this->filters.dims(1)*Cin); // comment to use nested for impl
      check_size(bias.dims(2), Cout, __func__);
      this->bias = bias;
    }

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
     */
    virtual ~Conv2d() {}

    /**
     * @brief Read in filters from a file given here if it wasn't passed to the constructor. Overwrites
     * current contents of this->filters.
     *
     * @param filters_filename The file where the filters tensor is saved. Will be loaded with torch.load(filename).
     * @param filt_dims The dimensions of the filter tensor in pytorch convention - (batch, channels, h, w)
     */
    inline void add_filters(const std::string &filters_filename,
                            const std::vector<int> &filt_dims){
      assert(filt_dims.size() > 0);
      _object *filts = utils("load_tensor", {pycpp::to_python(filters_filename)}, {});
      assert(filts);
      filters = from_numpy(reinterpret_cast<PyArrayObject *>(filts), filt_dims.size(), filt_dims);
      int Cout = filters.dims(3); int Cin = filters.dims(2);
      filters = af::unwrap(filters, params.filter_x, params.filter_y, params.filter_x, params.filter_y, 0, 0);
      filters = af::reorder(filters, 3, 0, 2, 1);
      filters = af::moddims(filters, Cout, filters.dims(1)*Cin); // comment to use nested for impl
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
      _object *bs = utils("load_tensor", {pycpp::to_python(bias_filename)}, {});
      assert(bs);
      bias = from_numpy(reinterpret_cast<PyArrayObject *>(bs), bias_dims.size(), bias_dims);
      check_size(bias.dims(2), filters.dims(0), __func__);
      this->has_bias = true;
    }

    /**
     * @brief Forward function, takes data and performs the Conv2d operation using the already-initialized
     * filters and bias tensors
     *
     * @param input Input data size (h_in, w_in, Cin, batch)
     * @return Convolved data size (h_out, w_out, Cout, batch)
     */
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::conv2d(params, input[0], filters, bias, this->has_bias)};
    }

    /**
     * @brief Forward function, takes data and performs the Conv2d operation using the already-initialized
     * filters and bias tensors
     *
     * @param input Input data size (h_in, w_in, Cin, batch)
     * @return Convolved data size (h_out, w_out, Cout, batch)
     */
    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::conv2d(params, input[0], filters, bias, this->has_bias)};
    }

  };
} // pytorch

#endif //PYTORCH_INFERENCE_CONVOLUTION_HPP
