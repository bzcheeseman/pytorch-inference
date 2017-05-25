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


#ifndef PYTORCH_INFERENCE_NORMALIZATION_IMPL_HPP
#define PYTORCH_INFERENCE_NORMALIZATION_IMPL_HPP

// STL
#include <assert.h>

// ArrayFire
#include <arrayfire.h>

namespace pytorch::impl {
  //! @todo: optimize this if possible?
  inline af::array batchnorm2d(const af::array &gamma, // vectors of size C (applied across channels)
                               const af::array &beta,
                               const af::array &running_mean,
                               const af::array &running_variance,
                               const float &epsilon,
                               const af::array &input){

    assert(gamma.dims(2) == input.dims(2));
    assert(beta.dims(2) == input.dims(2));
    int h_in = input.dims(0); int w_in = input.dims(1); int batch = input.dims(3);

    af::array out = (input - af::tile(running_mean, h_in, w_in, 1, batch))
                    /(af::sqrt(af::tile(running_variance, h_in, w_in, 1, batch)) + epsilon)
                    * af::tile(gamma, h_in, w_in, 1, batch) + af::tile(beta, h_in, w_in, 1, batch);

    return out;
  }
}

#endif //PYTORCH_INFERENCE_NORMALIZATION_IMPL_HPP
