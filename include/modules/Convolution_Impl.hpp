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

namespace pytorch::impl {
  /**
   * @brief Performs convolution given exported pytorch filters
   * @todo: optimize this
   *
   * @param filters (fh, fw, Cin, Cout)
   * @param bias (1, 1, Cout)
   * @param input (h, w, Cin, batch)
   * @return (h_out, w_out, Cout, batch)
   */
  inline af::array conv2d_old(const conv_params_t &params,
                              const af::array &input,
                              const af::array &filters,
                              const af::array &bias,
                              const bool &has_bias) {
    long Cin = input.dims(2);
    long Cout = filters.dims(0);
    long batch = input.dims(3);

    long h_in = input.dims(0);
    long w_in = input.dims(0);
    long h_out = (int) floor((h_in - params.filter_x + 2 * params.pad_x) / params.stride_x + 1);
    long w_out = (int) floor((w_in - params.filter_y + 2 * params.pad_y) / params.stride_y + 1);

    af::array out = af::constant(0, h_out, w_out, Cout, batch);  // (h, w, k, n)

    af::array in = af::unwrap(input, params.filter_x, params.filter_y,
                              params.stride_x, params.stride_y,
                              params.pad_x, params.pad_y);
    af::array b = af::constant(0, h_out, w_out, Cout, 1);
    if (has_bias)
      b = af::tile(bias, h_out, w_out, 1, 1) / (float) Cin;

    // input is (fx*fy, ox*oy, Cin, n)
    // filters is (Cout, fx*fy, Cin)
    // need to multiply each of Cout filters onto input
    // so take Cout x fx*fy . fx*fy x ox*oy = Cout x ox*oy

//#pragma omp parallel for collapse(2)
    for (int i = 0; i < batch; i++) {
      for (int k = 0; k < Cin; k++) {
        out(af::span, af::span, af::span, i) +=
                af::moddims(
                        af::reorder(
                                af::matmul(filters(af::span, af::span, k, 0), in(af::span, af::span, k, i)),
                                3, 1, 2, 0),
                        h_out, w_out, Cout) + b;
      }
    }

    return out;

  }

  /**
   * @brief Performs convolution given exported pytorch filters
   * @todo: optimize this
   *
   * @param filters (fh, fw, Cin, Cout)
   * @param bias (1, 1, Cout)
   * @param input (h, w, Cin, batch)
   * @return (h_out, w_out, Cout, batch)
   */
  inline af::array conv2d(const conv_params_t &params,
                          const af::array &input,
                          const af::array &filters,
                          const af::array &bias,
                          const bool &has_bias) { // why does this only sometimes work...

    long Cin = input.dims(2);
    long Cout = filters.dims(0);
    long batch = input.dims(3);

    long h_in = input.dims(0);
    long w_in = input.dims(0);
    long h_out = (int) floor((h_in - params.filter_x + 2 * params.pad_x) / params.stride_x + 1);
    long w_out = (int) floor((w_in - params.filter_y + 2 * params.pad_y) / params.stride_y + 1);

    af::array out = af::constant(0, h_out, w_out, Cout, batch);  // (h, w, k, n)

    af::array in = af::unwrap(input, params.filter_x, params.filter_y,
                              params.stride_x, params.stride_y,
                              params.pad_x, params.pad_y);
    in = af::moddims(in, params.filter_x*params.filter_y*Cin, in.dims(1), 1, in.dims(3));
    af::array b = af::constant(0, h_out, w_out, Cout, 1);
    if (has_bias)
      b = af::tile(bias, h_out, w_out, 1, 1);

    // input is (fx*fy, ox*oy, Cin, n)
    // filters is (Cout, fx*fy, Cin)
    // need to multiply each of Cout filters onto input
    // so take Cout x fx*fy*Cin . fx*fy*Cin x ox*oy = Cout x ox*oy

    for (int i = 0; i < batch; i++) {
      out(af::span, af::span, af::span, i) =
              af::moddims(
                      af::reorder(
                              af::matmul(filters(af::span, af::span, 0, 0), in(af::span, af::span, 0, i)),
                              3, 1, 2, 0),
                      h_out, w_out, Cout) + b;
    }

    return out;
  }
} // pytorch::impl



#endif //PYTORCH_INFERENCE_CONVOLUTION_IMPL_HPP
