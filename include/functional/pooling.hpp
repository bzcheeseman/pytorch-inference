//
// Created by Aman LaChapelle on 5/25/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_POOLING_IMPL_HPP
#define PYTORCH_INFERENCE_POOLING_IMPL_HPP

// STL
#include <iostream>

// ArrayFire
#include <arrayfire.h>

// Project
#include "../storage/tensor.hpp"

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

namespace pytorch::functional {
  inline tensor maxpool(const pooling_params_t &params,
                           const tensor &input,
                           tensor &indices){

    int h_in = input.data().dims(0); int w_in = input.data().dims(1);
    int h_out = (int)floor((h_in + 2*params.pad_x - (params.filter_x - 1) - 1)/params.stride_x + 1);
    int w_out = (int)floor((w_in + 2*params.pad_y - (params.filter_y - 1) - 1)/params.stride_y + 1);

    af::array in = af::unwrap(input.data(), params.filter_x, params.filter_y,
                              params.stride_x, params.stride_y,
                              params.pad_x, params.pad_y);

    af::array maxima, idx;
    af::max(maxima, idx, in, 0);
    af::array out = af::array(maxima, h_out, w_out, input.data().dims(2), input.data().dims(3));
    indices = idx;

    return tensor(out);

  }

  //! @todo: try to get this working with opencl
  inline tensor unpool(const pooling_params_t &params,
                          const tensor &input,
                          const tensor &indices){

    if (!(af::getActiveBackend() == AF_BACKEND_CPU)){
      std::cerr << "Unpooling not supported on OpenCL due to ArrayFire bug!" << std::endl;
      assert(af::getActiveBackend() == AF_BACKEND_CPU);
    }

    int h_in = input.data().dims(0); int w_in = input.data().dims(1);
    int h_out = (h_in - 1) * params.stride_x - 2*params.pad_x + params.filter_x;
    int w_out = (w_in - 1) * params.stride_y - 2*params.pad_y + params.filter_y;
    int C = input.data().dims(2); int batch = input.data().dims(3);

    af::array out = af::unwrap(af::constant(0, h_out, w_out, C, batch), params.filter_x, params.filter_y,
                               params.stride_x, params.stride_y,
                               params.pad_x, params.pad_y);

    af::array in, rowIdx, colIdx, temp;
    for (int i = 0; i < batch; i++){
      for (int j = 0; j < C; j++){
        in = af::moddims(input.data()(af::span, af::span, j, i), out.dims(1));
        colIdx = af::array(af::seq(out.dims(1))).as(s32);
        rowIdx = af::flat(indices.data()(0, af::span, j, i)).as(s32);
        temp = af::sparse(out.dims(0), out.dims(1), in, rowIdx, colIdx, AF_STORAGE_COO);
        out(af::span, af::span, j, i) = af::dense(temp);
      }
    }

    out = af::wrap(out, h_out, w_out, params.filter_x, params.filter_y,
                   params.stride_x, params.stride_y,
                   params.pad_x, params.pad_y);

    return out;
  }

  inline tensor avgpool(const pooling_params_t &params,
                           const tensor &input){
    int h_in = input.data().dims(0); int w_in = input.data().dims(1);
    int h_out = (int)floor((h_in + 2*params.pad_x - params.filter_x)/params.stride_x + 1);
    int w_out = (int)floor((w_in + 2*params.pad_y - params.filter_y)/params.stride_y + 1);

    af::array in = af::unwrap(input.data(), params.filter_x, params.filter_y,
                              params.stride_x, params.stride_y,
                              params.pad_x, params.pad_y);

    af::array means;
    means = af::mean(in, 0);
    af::array out = af::array(means, h_out, w_out, input.data().dims(2), input.data().dims(3));

    return out;
  }
} // pytorch::functional

#endif //PYTORCH_INFERENCE_POOLING_IMPL_HPP
