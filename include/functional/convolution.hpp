//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_CONVOLUTION_IMPL_HPP
#define PYTORCH_INFERENCE_CONVOLUTION_IMPL_HPP

// STL
#include <iostream>

// ArrayFire
#include <arrayfire.h>

// Project
#include "../utils.hpp"

namespace pytorch {
  /**
   * @struct conv_params_t
   * @file include/ops.hpp
   * @brief Holds the parameters needed for convolution as well as a convenience function to send them
   * to a std::ostream object.
   */
  struct conv_params_t {
    int filter_x, filter_y;
    int stride_x, stride_y;
    int pad_x, pad_y;

    friend inline std::ostream &operator<<(std::ostream &out, const conv_params_t &params){
      out << "filter_x: " << params.filter_x << " filter_y: " << params.filter_y << "\n";
      out << "stride_x: " << params.stride_x << " stride_y: " << params.stride_y << "\n";
      out << "pad_x: " << params.pad_x << " pad_y: " << params.pad_y << "\n";
      return out;
    }
  };
}

namespace pytorch::functional {
  /**
   * @brief Performs convolution given exported pytorch filters
   *
   * @param filters (fh, fw, Cin, Cout)
   * @param bias (1, 1, Cout)
   * @param input (h, w, Cin, batch)
   * @return (h_out, w_out, Cout, batch)
   */
  inline tensor conv2d(const conv_params_t &params,
                              const tensor &input,
                              const tensor &filters,
                              const tensor &bias,
                              const bool &has_bias) {
    long Cin = input.data().dims(2);
    long Cout = filters.data().dims(0);
    long batch = input.data().dims(3);

    long h_in = input.data().dims(0);
    long w_in = input.data().dims(0);
    long h_out = (int) floor((h_in - params.filter_x + 2 * params.pad_x) / params.stride_x + 1);
    long w_out = (int) floor((w_in - params.filter_y + 2 * params.pad_y) / params.stride_y + 1);

    af::array in = af::unwrap(input.data(), params.filter_x, params.filter_y,
                              params.stride_x, params.stride_y,
                              params.pad_x, params.pad_y);
    in = af::moddims(af::reorder(in, 0, 2, 1, 3), in.dims(0)*Cin, in.dims(1), 1, batch); // comment to use nested for functional
    af::array out = af::constant(0, Cout, h_out*w_out, batch, 1);
    if (has_bias)
      out += af::tile(af::reorder(bias.data(), 2, 3, 0, 1), 1, h_out*w_out, batch, 1);

    // input is (fx*fy, ox*oy, Cin, n)
    // filters is (Cout, fx*fy, Cin)
    // need to multiply each of Cout filters onto input
    // so take Cout x fx*fy*Cin . Cin*fx*fy x ox*oy = Cout x ox*oy

    for (int i = batch-1; i >= 0; i--) {
//      for (int k = 0; k < Cin; k++){ // faster for larger input planes (e.g. full size images)
//        out(af::span, af::span, i, 0) += af::matmul(filters(af::span, af::span, k, 0), in(af::span, af::span, k, i));
//      }
      out(af::span, af::span, i, 0) += af::matmul(filters.data()(af::span, af::span, 0, 0), in(af::span, af::span, 0, i));
    }

    return tensor(af::moddims(af::reorder(out, 1, 3, 0, 2), h_out, w_out, Cout, batch));
  }
} // pytorch::functional



#endif //PYTORCH_INFERENCE_CONVOLUTION_IMPL_HPP
