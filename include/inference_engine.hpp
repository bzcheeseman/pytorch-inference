//
// Created by Aman LaChapelle on 5/18/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_INFERENCE_ENGINE_HPP
#define PYTORCH_INFERENCE_INFERENCE_ENGINE_HPP

#include <algorithm>
#include <functional>

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
                     af::Backend backend = AF_BACKEND_CUDA,
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
          int wid = layer.size();
          for (int i = 0; i < wid; i++){
            out[i] = layer[i]->forward({out[i]})[0];
          }
        }
      }


      return out[0]; // must be a single tensor by the end
    }

  };
} // pytorch




#endif //PYTORCH_INFERENCE_INFERENCE_ENGINE_HPP
