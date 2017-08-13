//
// Created by Aman LaChapelle on 8/11/17.
//
// pytorch_inference
// Copyright (c) 2017 Aman LaChapelle
// Full license at pytorch_inference/LICENSE.txt
//

#ifndef PYTORCH_INFERENCE_RNN_IMPL_HPP
#define PYTORCH_INFERENCE_RNN_IMPL_HPP

// STL
#include <iostream>
#include <stdexcept>

// ArrayFire
#include <arrayfire.h>

// Project
#include "../storage/tensor.hpp"
#include "activations.hpp"

namespace pytorch {
  enum nonlinearity_t {
    RNN_TANH,
    RNN_RELU
  };

  struct rnn_params_t {
    nonlinearity_t nonlinearity = RNN_TANH;
    int seq_dim = 3;
  };
}

namespace pytorch::functional {
  inline tensor rnn(const rnn_params_t &params,
                          const tensor &input, // size = [input_size, 1, batch, seq_len]
                          const tensor &hidden, // size = [hidden_size, 1, 1, batch] <- for bidirectional/whatever in the future
                          const tensor &wih, // size = [hidden_size, input_size]
                          const tensor &bih, // size = [hidden_size]
                          const tensor &whh, // size = [hidden_size, hidden_size]
                          const tensor &bhh, // size = [hidden_size]
                          const bool &has_bias
  ) {

    af::array in;

    switch (params.seq_dim) {
      case 2 : in = af::reorder(input.data(), 0, 1, 3, 2); break; // swap last two dims
      case 3 : in = input.data(); // already in the way we want it
      default: throw std::runtime_error("Sequence dimension should be either 2 or 3");
    }

    tensor (*activation)(const tensor&);
    switch (params.nonlinearity) {
      case RNN_TANH : activation = &tanh; break;
      case RNN_RELU : activation = &relu; break;
    }

    long flat = in.dims(0) * in.dims(1);
    long batch = in.dims(2);
    long seq_len = in.dims(3);
    long hidden_size = hidden.data().dims(0);

    long comb_cols = wih.data().dims(1) + whh.data().dims(1); // should be input_size + hidden_size
    long comb_rows = wih.data().dims(0) + whh.data().dims(0); // should be 2 * hidden_size

    internal::check_size(wih.data().dims(1), flat, __func__);
    internal::check_size(whh.data().dims(1), hidden.data().dims(0), __func__);

    // Tile weight/bias to perform all matmuls at once
    af::array weight = af::constant(0.0, comb_rows, comb_cols, 1, 1);
    weight(af::seq(wih.data().dims(0)), af::seq(wih.data().dims(1)), 0, 0) = wih.data();
    weight(af::seq(wih.data().dims(0), comb_rows-1), af::seq(wih.data().dims(1), comb_cols-1), 0, 0) = whh.data();

    af::array bias = af::constant(0.0, 2 * hidden_size, batch);
    if (has_bias) {
      bias = af::tile(af::join(0, bih.data(), bhh.data()), 1, batch);
    }

    af::array out = af::constant(0.0, hidden_size, 1, seq_len, batch);

    for (int i = 0; i < seq_len; i++){ // apply the matmul over the sequence dimension
      af::array ih_t = af::join(0,
                                af::moddims(in(af::span, 0, af::span, i), flat, batch),
                                af::moddims(hidden.data(), hidden_size, batch)
      );
      out(af::span, 0, i, af::span) = activation(af::matmul(weight, ih_t) + bias).data();
    }

    return tensor(out);
  }
} // pytorch::functional


#endif //PYTORCH_INFERENCE_RNN_IMPL_HPP
