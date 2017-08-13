//
// Created by Aman LaChapelle on 8/12/17.
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


#ifndef PYTORCH_INFERENCE_TENSOR_HPP
#define PYTORCH_INFERENCE_TENSOR_HPP

#include <cassert>

#include <arrayfire.h>

#include "../py_object.hpp"
#include "../utils.hpp"

namespace pytorch {

  class tensor {
  private:
    af::array _data;
  public:
    tensor() = default;

    tensor(const af::array &arr) {
      this->_data = arr;
    }

    tensor(const tensor &other) = default;

    tensor(tensor &&other) = default;

    void load_numpy(const std::string &filename, std::vector<int> dims) {
      pycpp::py_object utils("utils", "../scripts");
      PyObject *array = utils("load_array", {pycpp::to_python(filename)}, {});
      assert(array);
      this->_data = internal::from_numpy(reinterpret_cast<PyArrayObject *>(array), dims.size(), dims);
    }

    void save(const std::string &filename, const std::string &key, const bool &append) {
      af::saveArray(key.c_str(), this->_data, filename.c_str(), append);
    }

    void load(const std::string &filename, const std::string &key) {
      this->_data = af::readArray(filename.c_str(), key.c_str());
    }

    af::array &data() { // now everyone can take a tensor argument without worrying about anything else.
      return this->_data;
    }

    af::array data() const { // now everyone can take a tensor argument without worrying about anything else.
      return this->_data;
    }

    af_array get() const {
      return this->_data.get();
    }

    void eval() {
      this->_data.eval();
    }

    tensor &operator=(const tensor &other) = default;

    tensor &operator=(const af::array &other) {
      this->_data = other;
      return *this;
    }

  };
} // pytorch


#endif //PYTORCH_INFERENCE_TENSOR_HPP
