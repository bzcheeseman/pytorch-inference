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

#include <arrayfire.h>

#include "utils.hpp"
#include "ops.hpp"
#include "layers.hpp"

namespace pytorch {

  /**
   * @class inference_engine
   * @file include/inference_engine.hpp
   * @brief The engine that drives inferences. This will hold everything related to the OpenCL backend.
   */
  class inference_engine {
  private:
    std::vector<Layer *> layers;
    const int device;

  public:
    inference_engine(const int &device = 0, af::Backend backend = AF_BACKEND_OPENCL) : device(device) {
      af::setBackend(backend);
      af::setDevice(device);
      af::info();
    }

    virtual ~inference_engine() {
      af::deviceGC();
    }

    inline void add_layer(Layer &l){
      layers.push_back(&l);
    }

    inline void add_layer(Layer *l){
      layers.push_back(l);
    }

    inline af::array forward(const af::array &input){
      af::array out (input);  // first input is input
      for (auto &l : layers){  // pass the data through the network
        auto temp = l->forward(out);
        out = af::array(temp);
      }
      af::sync(device);
      return out;
    }

  };
}  // pytorch




#endif //PYTORCH_INFERENCE_INFERENCE_ENGINE_HPP
