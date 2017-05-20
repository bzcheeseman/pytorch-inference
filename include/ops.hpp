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


#ifndef PYTORCH_INFERENCE_OPS_HPP
#define PYTORCH_INFERENCE_OPS_HPP

#include <arrayfire.h>
#include <assert.h>

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

  /**
   * @brief Performs convolution given exported pytorch filters
   * @param filters (fh, fw, Cin, Cout)
   * @param bias (1, 1, Cout)
   * @param input (h, w, Cin, batch)
   * @return (h_out, w_out, Cout, batch)
   */
  inline af::array conv2d(const conv_params_t &params,
                                const af::array &filters,
                                const af::array &bias,
                                const af::array &input){
    long Cin = input.dims(2); long Cout = filters.dims(3); long batch = input.dims(3);
    assert(bias.dims(2) == Cout);
    assert(filters.dims(2) == Cin);
    assert(filters.dims(0) == params.filter_x);
    assert(filters.dims(1) == params.filter_y);

    long h_in = input.dims(0); long w_in = input.dims(0);
    long h_out = (input.dims(0) - params.filter_x + 2*params.pad_x)/params.stride_x + 1;
    long w_out = (input.dims(1) - params.filter_y + 2*params.pad_y)/params.stride_y + 1;

    af::array out = af::constant(0, h_out, w_out, Cout, batch);  // (h, w, k, n)

    af::array in = af::unwrap(input, params.filter_x, params.filter_y,
                       params.stride_x, params.stride_y,
                       params.pad_x, params.pad_y);
    af::array f = af::reorder(filters, 3, 0, 1, 2);
    f = af::moddims(f, Cout, params.filter_x*params.filter_y, Cin);  // (Cout, h*w, Cin)
    af::array b = af::tile(bias, h_out, w_out, 1, 1);

    // input is (fx*fy, ox*oy, Cin, n)
    // filters is (Cout, fx*fy, Cin)
    // need to multiply each of Cout filters onto input
    // so take Cout x fx*fy . fx*fy x ox*oy = Cout x ox*oy

    // Once batched matmul is available, gotta get rid of these for loops (or just replace with gfor?)
#pragma omp parallel for  // parallelize the batch dimension, maybe we can replace in the future?
    for (int i = 0; i < batch; i++){
      for (int k = 0; k < Cin; k++){
        out(af::span, af::span, af::span, i) +=
                af::moddims(
                        af::reorder(
                                af::matmul(f(af::span, af::span, k, af::span), in(af::span, af::span, k, i)),
                        3, 1, 2, 0),
                h_out, w_out, Cout);
      }
      out(af::span, af::span, af::span, i) += b;
    }

    return out;

  }

  /**
   * @brief Performs the linear transformation y = Wx + b
   *
   * @param weight The weight matrix W
   * @param bias The bias vector b
   * @param input The input x
   * @return The transformed input (a.k.a. output) y
   */
  inline af::array linear(const af::array &weight, const af::array &bias, const af::array &input){
    long batch = input.dims(3);
    long flat = input.dims(0) * input.dims(1) * input.dims(2);

    assert(weight.dims(1) == flat);
    assert(bias.dims(0) == weight.dims(0));

    af::array out = af::tile(bias, 1, batch) + af::matmul(weight, af::moddims(af::transpose(input), flat, batch));
    out = af::reorder(out, 0, 3, 2, 1);

    return out;

  }

  inline std::vector<af::array> copy_branch(const af::array &input, const int &copies){
    assert(copies <= 4); // only 4 branches supported for now
    std::vector<af::array> out;
    for (int i = 0; i < copies; i++){
      out.push_back(input);
    }
    return out;
  }

  inline af::array cat2(const af::array &input1,
                         const af::array &input2,
                         const int &dim){
    return af::join(dim, input1, input2);
  }

  inline af::array cat3(const af::array &input1,
                        const af::array &input2,
                        const af::array &input3,
                        const int &dim){
    return af::join(dim, input1, input2, input3);
  }

  inline af::array cat4(const af::array &input1,
                        const af::array &input2,
                        const af::array &input3,
                        const af::array &input4,
                        const int &dim){
    return af::join(dim, input1, input2, input3, input4);
  }

  inline af::array sigmoid(const af::array &a){
    return 1 / (1 + af::exp(-a));
  }

  inline af::array hardtanh(const af::array &a, const float &low=1.f, const float &high=1.f){
    return af::clamp(a, low, high);  // just cuts it off at low/high (basicaly scaled hardtanh)?
  }

  inline af::array tanh(const af::array &a){
    return af::tanh(a);
  }

  inline af::array relu(const af::array &a){
    return af::max(a, af::constant(0, a.dims()));  // this should work
  }

  // TODO: MaxPool
  // TODO: BatchNorm (af::transform? Or just apply regular batchnorm?)

} // pytorch

#endif //PYTORCH_INFERENCE_OPS_HPP
