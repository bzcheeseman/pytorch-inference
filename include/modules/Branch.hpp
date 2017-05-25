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


#ifndef PYTORCH_INFERENCE_BRANCH_HPP
#define PYTORCH_INFERENCE_BRANCH_HPP

// Project
#include "Layer.hpp"
#include "Branch_Impl.hpp"

namespace pytorch {
  //! @todo: docs
  class Branch : public Layer {
  private:
    int copies;
  public:
    Branch(const int &copies) : copies(copies){}

    inline int get_copies() const {
      return copies;
    }

    inline std::vector<af::array> forward(const std::vector<af::array> &input){
      return impl::copy_branch(input[0], copies);
    }

    inline std::vector<af::array> operator()(const std::vector<af::array> &input){
      return impl::copy_branch(input[0], copies);
    }
  };
} // pytorch

#endif //PYTORCH_INFERENCE_BRANCH_HPP
