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


#ifndef PYTORCH_INFERENCE_POOLING_IMPL_HPP
#define PYTORCH_INFERENCE_POOLING_IMPL_HPP

// STL
#include <iostream>

// ArrayFire
#include <arrayfire.h>

namespace pytorch {
  /**
   * @struct conv_params_t
   * @file include/ops.hpp
   * @brief Holds the parameters needed for convolution as well as a convenience function to send them
   * to a std::ostream object.
   */
  struct pooling_params_t {
    int filter_x, filter_y;
    int stride_x, stride_y;
    int pad_x, pad_y;

    friend inline std::ostream &operator<<(std::ostream &out, const pooling_params_t &params){
      out << "filter_x: " << params.filter_x << " filter_y: " << params.filter_y << "\n";
      out << "stride_x: " << params.stride_x << " stride_y: " << params.stride_y << "\n";
      out << "pad_x: " << params.pad_x << " pad_y: " << params.pad_y << "\n";
      return out;
    }
  };
} // pytorch

namespace pytorch::impl {
  inline af::array maxpool(const pooling_params_t &params,
                           const af::array &input,
                           af::array &indices){

    int h_in = input.dims(0); int w_in = input.dims(1);
    int h_out = (int)floor((h_in + 2*params.pad_x - (params.filter_x - 1) - 1)/params.stride_x + 1);
    int w_out = (int)floor((w_in + 2*params.pad_y - (params.filter_y - 1) - 1)/params.stride_y + 1);

    af::array in = af::unwrap(input, params.filter_x, params.filter_y,
                              params.stride_x, params.stride_y,
                              params.pad_x, params.pad_y);

    af::array maxima, idx;
    af::max(maxima, idx, in, 0);
    af::array out = af::array(maxima, h_out, w_out, input.dims(2), input.dims(3));
    indices = idx;

    return out;

  }

  //! @todo: MUST OPTIMIZE THIS IS CRAP
  inline af::array unpool(const pooling_params_t &params,
                          const af::array &input,
                          const af::array &indices){

    int h_in = input.dims(0); int w_in = input.dims(1);
    int h_out = (h_in - 1) * params.stride_x - 2*params.pad_x + params.filter_x;
    int w_out = (w_in - 1) * params.stride_y - 2*params.pad_y + params.filter_y;
    int C = input.dims(2); int batch = input.dims(3);

    af::array out = af::unwrap(af::constant(0, h_out, w_out, C, batch), params.filter_x, params.filter_y,
                               params.stride_x, params.stride_y,
                               params.pad_x, params.pad_y);
    af::array in = af::unwrap(af::resize(input, h_out, w_out, AF_INTERP_LOWER), params.filter_x, params.filter_y,
                              params.stride_x, params.stride_y,
                              params.pad_x, params.pad_y);

    af::array idx;
#pragma #pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < batch; i++){ // there's gotta be a way to speed this up...
      for (int j = 0; j < C; j++){
        gfor (af::seq k, out.dims(1)){
          idx = indices(0, k, j, i).as(u32);
          out(idx.host<std::uint32_t>()[0], k, j, i) = in(idx.host<std::uint32_t>()[0], k, j, i);
        }
      }
    }

    out = af::wrap(out, h_out, w_out, params.filter_x, params.filter_y,
                   params.stride_x, params.stride_y,
                   params.pad_x, params.pad_y);

    return out;
  }

  inline af::array avgpool(const pooling_params_t &params,
                           const af::array &input){
    int h_in = input.dims(0); int w_in = input.dims(1);
    int h_out = (int)floor((h_in + 2*params.pad_x - params.filter_x)/params.stride_x + 1);
    int w_out = (int)floor((w_in + 2*params.pad_y - params.filter_y)/params.stride_y + 1);

    af::array in = af::unwrap(input, params.filter_x, params.filter_y,
                              params.stride_x, params.stride_y,
                              params.pad_x, params.pad_y);

    af::array means;
    means = af::mean(in, 0);
    af::array out = af::array(means, h_out, w_out, input.dims(2), input.dims(3));

    return out;
  }
} // pytorch::impl

#endif //PYTORCH_INFERENCE_POOLING_IMPL_HPP
