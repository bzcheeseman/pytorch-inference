//
// Created by Aman LaChapelle on 5/19/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_LAYERS_HPP
#define PYTORCH_INFERENCE_LAYERS_HPP

#include "modules/Layer.hpp"

#include "modules/Activations.hpp"
#include "modules/Branch.hpp"
#include "modules/Concatenate.hpp"
#include "modules/Convolution.hpp"
#include "modules/Linear.hpp"
#include "modules/Normalization.hpp"
#include "modules/Pooling.hpp"
#include "modules/Slice.hpp"
#include "modules/Sum.hpp"
#include "modules/Difference.hpp"
#include "modules/Product.hpp"
#include "modules/Divisor.hpp"

namespace pytorch {

  /**
   * @brief Convenience enum to use whenever you need to specify a dimension (like in Concat)
   */
  enum dims {
    n = 3,
    k = 2,
    h = 0,
    w = 1
  };

} // pytorch



#endif //PYTORCH_INFERENCE_LAYERS_HPP
