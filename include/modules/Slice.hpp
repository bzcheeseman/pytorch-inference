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


#ifndef PYTORCH_INFERENCE_SLICE_HPP
#define PYTORCH_INFERENCE_SLICE_HPP

#include "Layer.hpp"
#include "Slice_Impl.hpp"

namespace pytorch {
  //! @todo: docs
  class Slice2 : public Layer {
  private:
    int dim;
  public:
    Slice2(const int &dim) : dim(dim) {}

    inline int get_dim() const {
      return dim;
    }

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return impl::split_branch(input[0], 2, dim);
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return impl::split_branch(input[0], 2, dim);
    }

  };

//! @todo: docs
  class Slice3 : public Layer {
  private:
    int dim;
  public:
    Slice3(const int &dim) : dim(dim) {}

    inline int get_dim() const {
      return dim;
    }

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return impl::split_branch(input[0], 3, dim);
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return impl::split_branch(input[0], 3, dim);
    }

  };

//! @todo: docs
  class Slice4 : public Layer {
  private:
    int dim;
  public:
    Slice4(const int &dim) : dim(dim) {}

    inline int get_dim() const {
      return dim;
    }

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return impl::split_branch(input[0], 4, dim);
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return impl::split_branch(input[0], 4, dim);
    }

  };
}

#endif //PYTORCH_INFERENCE_SLICE_HPP
