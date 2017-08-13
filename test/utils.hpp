//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//


#ifndef PYTORCH_INFERENCE_UTILS_HPP
#define PYTORCH_INFERENCE_UTILS_HPP

// Python
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// STL
#include <assert.h>
#include <vector>

// ArrayFire
#include <arrayfire.h>

// Project
#include "../include/storage/tensor.hpp"
#include "../include/utils.hpp"
#include "../include/py_object.hpp"

inline std::vector<pytorch::tensor> test_setup(const std::vector<int> &n,
                                         const std::vector<int> &k,
                                         const std::vector<int> &h,
                                         const std::vector<int> &w,
                                         const std::vector<int> &out_n,
                                         const std::vector<int> &out_k,
                                         const std::vector<int> &out_h,
                                         const std::vector<int> &out_w,
                                         const std::vector<std::string> &save_file,
                                         const std::string &python_function){

  pycpp::py_object utils ("utils", "../scripts");
  pycpp::py_object py_function (python_function, "../scripts");
  assert(n.size() == k.size() && k.size() == h.size() && h.size() == w.size() && w.size() == save_file.size());
  std::vector<pytorch::tensor> out;
  int n_tensors = n.size();
  std::vector<PyObject *> python_func_args;
  for (int i = 0; i < n_tensors; i++){
    utils("save_tensor", {pycpp::to_python(n[i]),
                         pycpp::to_python(k[i]),
                         pycpp::to_python(h[i]),
                         pycpp::to_python(w[i]),
                         pycpp::to_python(save_file[i])});
    python_func_args.push_back(pycpp::to_python(save_file[i]));

    PyObject *tensor = utils("load_tensor", {pycpp::to_python(save_file[i])}, {});
    out.push_back(pytorch::internal::from_numpy(reinterpret_cast<PyArrayObject *>(tensor), 4,
                                      {n[i], k[i], h[i], w[i]}));

  }

  PyObject *args = pycpp::make_tuple(python_func_args);

  af::timer::start();
  PyObject *pto = py_function(python_function, args, NULL);
  std::cout << "pytorch forward took (s): " << af::timer::stop() << std::endl;

  assert(pto);

  if (PyList_Check(pto)){
    long len = PyList_Size(pto);
    for (int i = 0; i < len; i++){
      PyObject *item = PyList_GetItem(pto, i);
      out.push_back(pytorch::tensor(pytorch::internal::from_numpy(reinterpret_cast<PyArrayObject *>(item), 4,
                                        {out_n[i], out_k[i], out_h[i], out_w[i]})));
    }
  }
  else{ // if it isn't a list then it's one tensor
    out.push_back(pytorch::tensor(pytorch::internal::from_numpy(reinterpret_cast<PyArrayObject *>(pto), 4,
                                      {out_n[0], out_k[0], out_h[0], out_w[0]})));
  }

  return out; // out has {input_tensors, pytorch_output_tensors} in that order exactly

}

inline bool almost_equal(const af::array &first, const af::array &second,
                         const float &epsilon=std::numeric_limits<float>::epsilon()*500){
  af::array condition = af::flat(af::abs(first - second) <= epsilon);
  if (af::allTrue<bool>(condition)){
    return true;
  }
  else{
    af_print(first); af_print(second); af_print(first - second);
    return false;
  }
}

inline bool almost_equal(const pytorch::tensor &first, const pytorch::tensor &second,
                         const float &epsilon=std::numeric_limits<float>::epsilon()*500){
  af::array condition = af::flat(af::abs(first.data() - second.data()) <= epsilon);
  if (af::allTrue<bool>(condition)){
    return true;
  }
  else{
    af_print(first.data()); af_print(second.data()); af_print(first.data() - second.data());
    return false;
  }
}

#endif //PYTORCH_INFERENCE_UTILS_HPP
