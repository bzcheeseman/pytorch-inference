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


#ifndef PYTORCH_INFERENCE_CONCATENATE_HPP
#define PYTORCH_INFERENCE_CONCATENATE_HPP

#include "Layer.hpp"
#include "Concatenate_Impl.hpp"

namespace pytorch {

  class Concat : public Layer {
    int dim;
    int n_tensors;
  public:
    Concat(const int &dim, const int &n_tensors) : dim(dim), n_tensors(n_tensors) {}

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return {impl::catn(input, dim)};
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return {impl::catn(input, dim)};
    }
  };

} // pytorch

#endif //PYTORCH_INFERENCE_CONCATENATE_HPP
