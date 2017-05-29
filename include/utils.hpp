//
// Created by Aman LaChapelle on 5/18/17.
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


#ifndef PYTORCH_INFERENCE_EXTRACT_NUMPY_HPP
#define PYTORCH_INFERENCE_EXTRACT_NUMPY_HPP

// Python
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// STL
#include <stdexcept>

// ArrayFire
#include <arrayfire.h>
#include <future>

namespace pytorch {

  /**
   * @brief Converts a numpy array to an ArrayFire array
   *
   * @param array numpy ndarray (PyArrayObject *) object
   * @param ndim Number of dimensions, usually 4
   * @param dims The array dimensions - note these are in the torch convention (n, k, h, w), we will convert to
   *             ArrayFire convention which is (h, w, k, n) in the output array. This will be obscured as much as possible.
   * @return ArrayFire array that has the data from the numpy array arranged within.
   */
  inline af::array from_numpy(PyArrayObject *array, int ndim, std::vector<int> dims){

    array = PyArray_GETCONTIGUOUS(array);  // make sure it's contiguous (might already be)

    int array_ndim = PyArray_NDIM(array);
    assert(ndim == array_ndim);

    npy_intp *array_dims = PyArray_SHAPE(array);

    for (int i = 0; i < ndim; i++){
      assert(dims[i] == array_dims[i]);  // make sure dimensions are right
    }

    int n, k, h, w;
    n = dims[0]; k = dims[1]; h = dims[2]; w = dims[3];

    af::array out (w, h, k, n, reinterpret_cast<float *>(PyArray_DATA(array)), afHost);  // errors out here
    out = af::reorder(out, 1, 0, 2, 3);  // reorder to arrayfire specs (h, w, k, batch)

    return out;

  }

  inline void check_size(const int &size1, const int &size2, const std::string &func){
    if (size1 == size2){
      return;
    }
    else{
      std::string error = "Incorrect size passed! Sizes: " + std::to_string(size1) + ", " + std::to_string(size2);
      error += " Function: " + func;
      throw std::runtime_error(error);
    }
  }

  inline void check_num_leq(const int &size1, const int &size2, const std::string &func){
    if (size1 <= size2){
      return;
    }
    else{
      std::string error = "Incorrect size passed! Sizes: " + std::to_string(size1) + ", " + std::to_string(size2);
      error += " Function: " + func;
      throw std::runtime_error(error);
    }
  }

  template<class Layer>
  inline std::vector<std::future<af::array>> async_for(std::vector<Layer *> f, std::vector<af::array> inputs){
    std::vector<std::future<af::array>> out;
    int n_out = f.size();
    for (int i = 0; i < n_out; i++){
      out.push_back(std::async(std::launch::async, &Layer::forward, f[i], inputs[i]));
    }
    return out;
  }

} // pytorch

#endif //PYTORCH_INFERENCE_EXTRACT_NUMPY_HPP
