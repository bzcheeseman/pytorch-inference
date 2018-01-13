//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_NORMALIZATION_HPP
#define PYTORCH_INFERENCE_NORMALIZATION_HPP

// STL
#include <assert.h>
#include <string>
#include <vector>

// ArrayFire
#include <arrayfire.h>

// Project
#include "Layer.hpp"
#include "../functional/normalization.hpp"
#include "../py_object.hpp"
#include "../utils.hpp"

namespace pytorch {
  /**
   * @class BatchNorm2d "include/layers.hpp"
   * @file "include/layers.hpp"
   * @brief Equivalent to BatchNorm2d in pytorch.
   *
   * Implements the forward pass for pytorch's nn.BatchNorm2d module. Note that you do need to extract the proper
   * running mean, running variance, gamma and beta tensors from pytorch.
   */
  class BatchNorm2d : public Layer {
  private:
    tensor gamma;
    tensor beta;
    tensor running_mean;
    tensor running_var;
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
    BatchNorm2d(const tensor &gamma,
                const tensor &beta,
                const tensor &running_mean,
                const tensor &running_var,
                const float &epsilon = 1e-5) : gamma(gamma), beta(beta),
                                               running_mean(running_mean), running_var(running_var),
                                               epsilon(epsilon) {}

    /**
     * @brief Constructs a BatchNorm2d object and loads the requisite tensors in from filenames and sizes.
     *
     * @param gamma_filename The file where gamma can be found. Will be loaded with numpy.load(filename).
     * @param gamma_dims The dimensions of gamma in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     * @param beta_filename The file where beta can be found. Will be loaded with numpy.load(filename).
     * @param beta_dims The dimensions of beta in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     * @param running_mean_filename The file where running_mean can be found. Will be loaded with numpy.load(filename).
     * @param running_mean_dims The dimensions of running_mean in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     * @param running_var_filename the file where running_var can be found. Will be loaded with numpy.load(filename).
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
     * @brief Default destructor - may need some more functionality.
     * TODO: add filter/bias saving on destroy, corresponding constructor to remake layer
     */
    virtual ~BatchNorm2d() = default;

    /**
     * @brief Adds gamma to the layer if the name wasn't passed to the constructor
     *
     * @param gamma_filename The file where gamma can be found. Will be loaded with numpy.load(filename).
     * @param gamma_dims The dimensions of gamma in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     */
    inline void add_gamma(const std::string &gamma_filename = "",
                          const std::vector<int> &gamma_dims = {}){
      assert(gamma_dims.size() > 0);
      PyObject *g = utils("load_array", {pycpp::to_python(gamma_filename)}, {});
      assert(g);
      gamma = internal::from_numpy(reinterpret_cast<PyArrayObject *>(g), gamma_dims.size(), gamma_dims);
    }

    /**
     * @brief Adds beta if it wasn't added by the constructor.
     *
     * @param beta_filename The file where beta can be found. Will be loaded with numpy.load(filename).
     * @param beta_dims The dimensions of beta in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     */
    inline void add_beta(const std::string &beta_filename = "",
                         const std::vector<int> &beta_dims = {}){
      assert(beta_dims.size() > 0);
      PyObject *b = utils("load_array", {pycpp::to_python(beta_filename)}, {});
      assert(b);
      beta = internal::from_numpy(reinterpret_cast<PyArrayObject *>(b), beta_dims.size(), beta_dims);
    }

    /**
     * @brief Adds running_mean if it wasn't added by the constructor.
     *
     * @param running_mean_filename The file where running_mean can be found. Will be loaded with numpy.load(filename).
     * @param running_mean_dims The dimensions of running_mean in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     */
    inline void add_running_mean(const std::string &running_mean_filename = "",
                                 const std::vector<int> &running_mean_dims = {}){
      assert(running_mean_dims.size() > 0);
      PyObject *rm = utils("load_array", {pycpp::to_python(running_mean_filename)}, {});
      assert(rm);
      running_mean = internal::from_numpy(reinterpret_cast<PyArrayObject *>(rm), running_mean_dims.size(), running_mean_dims);
    }

    /**
     * @brief Adds running_var if it wasn't added by the constructor.
     *
     * @param running_var_filename the file where running_var can be found. Will be loaded with numpy.load(filename).
     * @param running_var_dims The dimensions of running_var in pytorch convention (n, k, h, w) (usually = (1, k, 1, 1))
     */
    inline void add_running_var(const std::string &running_var_filename = "",
                                const std::vector<int> &running_var_dims = {}){
      assert(running_var_dims.size() > 0);
      PyObject *rv = utils("load_array", {pycpp::to_python(running_var_filename)}, {});
      assert(rv);
      running_var = internal::from_numpy(reinterpret_cast<PyArrayObject *>(rv), running_var_dims.size(), running_var_dims);
    }

    /**
     * @brief Applies the forward pass of batch normalization
     *
     * @param input The input data to be normalized.
     * @return The normalized data. The size has not changed.
     */
    inline std::vector<tensor> forward(const std::vector<tensor> &input){
      return {functional::batchnorm2d(gamma, beta, running_mean, running_var, epsilon, input[0])};
    }

    /**
     * @brief Applies the forward pass of batch normalization
     *
     * @param input The input data to be normalized.
     * @return The normalized data. The size has not changed.
     */
    inline std::vector<tensor> operator()(const std::vector<tensor> &input){
      return {functional::batchnorm2d(gamma, beta, running_mean, running_var, epsilon, input[0])};
    }

  };
} // pytorch

#endif //PYTORCH_INFERENCE_NORMALIZATION_HPP
