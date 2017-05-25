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


#ifndef PYTORCH_INFERENCE_LAYER_HPP_HPP
#define PYTORCH_INFERENCE_LAYER_HPP_HPP

// STL
#include <vector>

// ArrayFire
#include <arrayfire.h>

namespace pytorch {
  /**
   * @class Layer
   * @file "include/layers.hpp"
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
   * @class Skip "include/layers.hpp"
   * @file "include/layers.hpp"
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
} // pytorch

#endif //PYTORCH_INFERENCE_LAYER_HPP_HPP
