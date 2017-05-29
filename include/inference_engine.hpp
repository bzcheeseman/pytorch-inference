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


#ifndef PYTORCH_INFERENCE_INFERENCE_ENGINE_HPP
#define PYTORCH_INFERENCE_INFERENCE_ENGINE_HPP

#include <fstream>
#include <sstream>
#include <cstdint>
#include <vector>
#include <algorithm>

#include <arrayfire.h>

#include "utils.hpp"
#include "layers.hpp"
#include "layers.hpp"

namespace pytorch {

  /**
   * @class inference_engine
   * @file include/inference_engine.hpp
   * @brief The engine that drives inferences. This will hold everything related to the OpenCL backend.
   */
  class inference_engine {
  private:
    std::vector<std::vector<pytorch::Layer *>> layers;  // how to speed this up to avoid vtable lookup?
    const int device;

  public:
    inference_engine(const int &device = 0,
                     af::Backend backend = AF_BACKEND_OPENCL,
                     bool quiet = true) : device(device) {
      af::setBackend(backend);
      af::setDevice(this->device);
      if (!quiet){
        af::info();
      }
    }

    virtual ~inference_engine() {
      af::deviceGC();
    }

    inline void add_layer(Layer *l){
      layers.push_back({l});
    }

    inline void add_layer(std::vector<Layer *>l){
      layers.push_back(l);
    }

    inline Layer *get_layer_ptr(const int &depth, const int &width = 0){
      return layers[depth][width];
    }

    inline af::array forward(const std::vector<af::array> &input){
      std::vector<af::array> out = input;
      check_num_leq(out.size(), 10, __func__); // there are checks in each layer to make sure it's less than 10
      for (auto &layer : layers){
        if (layer.size() == 1) {
          out = layer[0]->forward(out);
        }
        else{
          check_size(out.size(), layer.size(), __func__); // make sure there are enough inputs
          std::transform(out.begin(), out.end(), out.begin(),
                         [&](const af::array &a) -> af::array {
                           return layer[&a - &out[0]]->forward({a})[0];
                         });
        }
      }


      return out[0]; // must be a single tensor by the end
    }

  };
} // pytorch




#endif //PYTORCH_INFERENCE_INFERENCE_ENGINE_HPP
