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
#include "Normalization_Impl.hpp"
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
      PyObject *g = utils("load_tensor", {pycpp::to_python(gamma_filename)}, {});
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
      PyObject *b = utils("load_tensor", {pycpp::to_python(beta_filename)}, {});
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
      PyObject *rm = utils("load_tensor", {pycpp::to_python(running_mean_filename)}, {});
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
      PyObject *rv = utils("load_tensor", {pycpp::to_python(running_var_filename)}, {});
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
      return {impl::batchnorm2d(gamma, beta, running_mean, running_var, epsilon, input[0])};
    }

    /**
     * @brief Applies the forward pass of batch normalization
     *
     * @param input The input data to be normalized.
     * @return The normalized data. The size has not changed.
     */
    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::batchnorm2d(gamma, beta, running_mean, running_var, epsilon, input[0])};
    }

  };
} // pytorch

#endif //PYTORCH_INFERENCE_NORMALIZATION_HPP
