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
    bool has_bias;

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
           const bool &has_bias = false,
           const std::string &bias_filename = "",
           const std::vector<int> &bias_dims = {},
           const std::string &python_home = "../scripts") : params(params),
                                                            utils("utils", python_home),
                                                            has_bias(has_bias) {

      if (!filters_filename.empty()){
        this->add_filters(filters_filename, filt_dims);
      }

      if (!bias_filename.empty() && has_bias){
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
      return {conv2d(params, input[0], filters, bias, has_bias)};
    }

    /**
     * @brief Forward function, takes data and performs the Conv2d operation using the already-initialized
     * filters and bias tensors
     *
     * @param input Input data size (h_in, w_in, Cin, batch)
     * @return Convolved data size (h_out, w_out, Cout, batch)
     */
    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {conv2d(params, input[0], filters, bias, has_bias)};
    }

  };

  /**
   * @class MaxPool2d (TESTED)
   * @file include/layers.hpp
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

    /**
     * @brief Implements the forward pass
     *
     * @param input The input array to be pooled
     * @return The pooled array
     */
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {pytorch::maxpool(params, input[0], indices)};
    }

    /**
     * @brief Implements the forward pass
     *
     * @param input The input array to be pooled
     * @return The pooled array
     */
    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {pytorch::maxpool(params, input[0], indices)};
    }
  };

  /**
   * @class AvgPool2d (TESTED)
   * @file include/layers.hpp
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
      return {pytorch::avgpool(params, input[0])};
    }

    /**
     * @brief Implements the forwards pass
     *
     * @param input The input array to be pooled
     * @return The pooled array
     */
    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {pytorch::avgpool(params, input[0])};
    }
  };

  /**
   * @class BatchNorm2d (TESTED)
   * @file include/layers.hpp
   * @brief Equivalent to BatchNorm2d in pytorch.
   *
   * Implements the forward pass for pytorch's nn.BatchNorm2d module. Note that you do need to extract the proper
   * running mean, running variance, gamma and beta tensors from pytorch.
   */
  class BatchNorm2d : public Layer {
  private:
    af::array gamma;
    af::array beta;
    af::array running_mean;
    af::array running_var;
    float epsilon;
    pycpp::py_object utils;
  public:
    /**
     * @brief Constructs a BatchNorm2d object.
     *
     * @param gamma The multiplier for the affine transform. Saved as 'bn.weight' by pytorch.
     * @param beta The bias for the affine transform. Saved as 'bn.bias' by pytorch.
     * @param running_mean The running mean for the batchnorm operation.
     * @param running_var The running variance for the batchnorm operation.
     * @param epsilon A factor in the denominator of the transform that adds stability.
     */
    BatchNorm2d(const af::array &gamma,
                const af::array &beta,
                const float &running_mean,
                const float &running_var,
                const float &epsilon = 1e-5) : gamma(gamma), beta(beta),
                                             running_mean(running_mean), running_var(running_var),
                                             epsilon(epsilon) {}

    /**
     * @brief Constructs a BatchNorm2d object and loads the requisite tensors in from filenames and sizes.
     *
     * @param gamma_filename The file where gamma can be found. Will be loaded with torch.load(filename).
     * @param gamma_dims The dimensions of gamma in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     * @param beta_filename The file where beta can be found. Will be loaded with torch.load(filename).
     * @param beta_dims The dimensions of beta in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     * @param running_mean_filename The file where running_mean can be found. Will be loaded with torch.load(filename).
     * @param running_mean_dims The dimensions of running_mean in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     * @param running_var_filename the file where running_var can be found. Will be loaded with torch.load(filename).
     * @param running_var_dims The dimensions of running_var in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     * @param epsilon A float for numerical stability, 1e-5 by default.
     * @param python_home Where the utility scripts are - holds the loading script necessary to load up the tensors.
     */
    BatchNorm2d(const std::string &gamma_filename = "",
                const std::vector<int> &gamma_dims = {},
                const std::string &beta_filename = "",
                const std::vector<int> &beta_dims = {},
                const std::string &running_mean_filename = "",
                const std::vector<int> &running_mean_dims = {},
                const std::string &running_var_filename = "",
                const std::vector<int> &running_var_dims = {},
                const float &epsilon = 1e-5,
                const std::string &python_home = "../scripts") : utils("utils", python_home), epsilon(epsilon){

      if (!gamma_filename.empty()){
        this->add_gamma(gamma_filename, gamma_dims);
      }
      if (!beta_filename.empty()){
        this->add_beta(beta_filename, beta_dims);
      }
      if (!running_mean_filename.empty()){
        this->add_running_mean(running_mean_filename, running_mean_dims);
      }
      if (!running_var_filename.empty()){
        this->add_running_var(running_var_filename, running_var_dims);
      }

    }

    /**
     * @brief Adds gamma to the layer if the name wasn't passed to the constructor
     *
     * @param gamma_filename The file where gamma can be found. Will be loaded with torch.load(filename).
     * @param gamma_dims The dimensions of gamma in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     */
    inline void add_gamma(const std::string &gamma_filename = "",
                          const std::vector<int> &gamma_dims = {}){
      assert(gamma_dims.size() > 0);
      PyObject *g = utils("load_tensor", {pycpp::to_python(gamma_filename)});
      assert(g);
      gamma = from_numpy(reinterpret_cast<PyArrayObject *>(g), gamma_dims.size(), gamma_dims);
    }

    /**
     * @brief Adds beta if it wasn't added by the constructor.
     *
     * @param beta_filename The file where beta can be found. Will be loaded with torch.load(filename).
     * @param beta_dims The dimensions of beta in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     */
    inline void add_beta(const std::string &beta_filename = "",
                          const std::vector<int> &beta_dims = {}){
      assert(beta_dims.size() > 0);
      PyObject *b = utils("load_tensor", {pycpp::to_python(beta_filename)});
      assert(b);
      beta = from_numpy(reinterpret_cast<PyArrayObject *>(b), beta_dims.size(), beta_dims);
    }

    /**
     * @brief Adds running_mean if it wasn't added by the constructor.
     *
     * @param running_mean_filename The file where running_mean can be found. Will be loaded with torch.load(filename).
     * @param running_mean_dims The dimensions of running_mean in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     */
    inline void add_running_mean(const std::string &running_mean_filename = "",
                          const std::vector<int> &running_mean_dims = {}){
      assert(running_mean_dims.size() > 0);
      PyObject *rm = utils("load_tensor", {pycpp::to_python(running_mean_filename)});
      assert(rm);
      running_mean = from_numpy(reinterpret_cast<PyArrayObject *>(rm), running_mean_dims.size(), running_mean_dims);
    }

    /**
     * @brief Adds running_var if it wasn't added by the constructor.
     *
     * @param running_var_filename the file where running_var can be found. Will be loaded with torch.load(filename).
     * @param running_var_dims The dimensions of running_var in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     */
    inline void add_running_var(const std::string &running_var_filename = "",
                                 const std::vector<int> &running_var_dims = {}){
      assert(running_var_dims.size() > 0);
      PyObject *rv = utils("load_tensor", {pycpp::to_python(running_var_filename)});
      assert(rv);
      running_var = from_numpy(reinterpret_cast<PyArrayObject *>(rv), running_var_dims.size(), running_var_dims);
    }

    /**
     * @brief Applies the forward pass of batch normalization
     *
     * @param input The input data to be normalized.
     * @return The normalized data. The size has not changed.
     */
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {batchnorm2d(gamma, beta, running_mean, running_var, epsilon, input[0])};
    }

    /**
     * @brief Applies the forward pass of batch normalization
     *
     * @param input The input data to be normalized.
     * @return The normalized data. The size has not changed.
     */
    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {batchnorm2d(gamma, beta, running_mean, running_var, epsilon, input[0])};
    }

  };

  /**
   * @class Linear (TESTED)
   * @file include/layers.hpp
   * @brief Equivalent to Linear in pytorch
   *
   * Implements the forward pass for pytorch's nn.Linear module. Note that clearly something needs to happen to
   * get the tensors from python to C++, but I've tried to take care of this through the import constructor.
   */
  class Linear : public Layer {
  private:
    af::array weights;
    af::array bias;
    pycpp::py_object utils;
    bool has_bias;

  public:
    /**
     * @brief Constructs a Linear object given weights, and bias tensors.

     * @param weights The trained weight tensors. For those comfortable with Py_Cpp.
     * @param bias The trained bias tensors. For those comfortable with Py_Cpp. Can be initialized to zero.
     */
    Linear(const af::array &weights,
           const af::array &bias) : weights(weights), bias(bias) { }

    /**
     * @brief Constructs a Linear object given the filenames and sizes of the requisite tensors.
     *
     * @param weights_filename The file where the weights tensor is saved. Will be loaded with torch.load(filename).
     * @param weights_dims The dimensions of the weights tensor in pytorch convention - (batch, channels, h, w)
     * @param bias_filename The file where the bias tensor is saved. Will be loaded with torch.load(filename).
     * @param bias_dims The dimensions of the bias tensor in pytorch convention - (batch, channels, h, w)
     * @param python_home Where the utility scripts are - holds the loading script necessary to load up the tensors.
     */
    Linear(const std::string &weights_filename = "",
           const std::vector<int> &weights_dims = {},
           const bool has_bias = false,
           const std::string &bias_filename = "",
           const std::vector<int> &bias_dims = {},
           const std::string &python_home = "../scripts") : utils("utils", python_home), has_bias(has_bias) {

      if (!weights_filename.empty()){
        this->add_weights(weights_filename, weights_dims);
      }
      if (!bias_filename.empty() && has_bias){
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
     */
    virtual ~Linear() {}

    /**
     * @brief Read in weights from a file given here if it wasn't passed to the constructor. Overwrites
     * current contents of this->weights.
     *
     * @param weights_filename The file where the weights tensor is saved. Will be loaded with torch.load(filename).
     * @param weights_dims The dimensions of the weights tensor in pytorch convention - (batch, channels, h, w)
     */
    inline void add_weights(const std::string &weights_filename,
                            const std::vector<int> &weights_dims){
      assert(weights_dims.size() > 0);
      PyObject *ws = utils("load_tensor", {pycpp::to_python(weights_filename)});
      assert(ws);
      weights = from_numpy(reinterpret_cast<PyArrayObject *>(ws), weights_dims.size(), weights_dims);
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
      this->has_bias = true;
      PyObject *bs = utils("load_tensor", {pycpp::to_python(bias_filename)});
      assert(bs);
      bias = from_numpy(reinterpret_cast<PyArrayObject *>(bs), bias_dims.size(), bias_dims);
    }

    /**
     * @brief Sets whether or not this layer has bias. If no, then this should be called. Otherwise, it's unnecessary.
     * @param has_bias Whether or not the layer has bias.
     */
    inline void set_has_bias(bool has_bias){
      this->has_bias = has_bias;
    }

    /**
     * @brief Forward function, takes data and performs the Linear operation using the already-initialized
     * weights and bias tensors
     *
     * @param input Input data size (dims_in, 1, 1, batch)
     * @return Transformed data size (dims_out, 1, batch)
     */
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {linear(input[0], weights, bias, this->has_bias)};
    }

    /**
     * @brief Forward function, takes data and performs the Linear operation using the already-initialized
     * weights and bias tensors
     *
     * @param input Input data size (dims_in, 1, 1, batch)
     * @return Transformed data size (dims_out, 1, batch)
     */
    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {linear(input[0], weights, bias, this->has_bias)};
    }

  };

  //! @todo: add documentation for the rest of the classes

  /* Branch - tested */
  class Branch : public Layer {
  private:
    int copies;
  public:
    Branch(const int &copies) : copies(copies){}

    inline int get_copies() const {
      return copies;
    }

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return pytorch::copy_branch(input[0], copies);
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return pytorch::copy_branch(input[0], copies);
    }
  };

  class Slice2 : public Layer {
  private:
    int dim;
  public:
    Slice2(const int &dim) : dim(dim) {}

    inline int get_dim() const {
      return dim;
    }

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return pytorch::split_branch(input[0], 2, dim);
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return pytorch::split_branch(input[0], 2, dim);
    }

  };

  class Slice3 : public Layer {
  private:
    int dim;
  public:
    Slice3(const int &dim) : dim(dim) {}

    inline int get_dim() const {
      return dim;
    }

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return pytorch::split_branch(input[0], 3, dim);
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return pytorch::split_branch(input[0], 3, dim);
    }

  };

  class Slice4 : public Layer {
  private:
    int dim;
  public:
    Slice4(const int &dim) : dim(dim) {}

    inline int get_dim() const {
      return dim;
    }

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return pytorch::split_branch(input[0], 4, dim);
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return pytorch::split_branch(input[0], 4, dim);
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

  // Softmax is not a stable operation so it's making testing hard...
  class Softmax : public Layer { // SO SLOW GOOD LORD
  public:
    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {pytorch::softmax(input[0])};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {pytorch::softmax(input[0])};
    }
  };

} // pytorch



#endif //PYTORCH_INFERENCE_LAYERS_HPP
