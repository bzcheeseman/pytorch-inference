#include <iostream>

#include "include/utils.hpp"
#include "include/py_object.hpp"
#include "include/ops.hpp"
#include "include/layers.hpp"
#include "include/inference_engine.hpp"

int main() {

  // Set up data - this just creates random tensors in the specified shapes
  pycpp::python_home = "../scripts";
  pycpp::py_object test ("test");
  test("save_tensor", {pycpp::to_python(16), // out_filters
                       pycpp::to_python(3), // in_filters
                       pycpp::to_python(3),
                       pycpp::to_python(1),
                       pycpp::to_python("filts.dat")});

  test("save_tensor", {pycpp::to_python(1),
                       pycpp::to_python(16), // out_filters
                       pycpp::to_python(1),
                       pycpp::to_python(1),
                       pycpp::to_python("bias.dat")});

  test("save_tensor", {pycpp::to_python(1),
                       pycpp::to_python(16),
                       pycpp::to_python(1),
                       pycpp::to_python(1),
                       pycpp::to_python("gamma.dat")});

  test("save_tensor", {pycpp::to_python(1),
                       pycpp::to_python(16),
                       pycpp::to_python(1),
                       pycpp::to_python(1),
                       pycpp::to_python("beta.dat")});

  test("save_tensor", {pycpp::to_python(1),
                       pycpp::to_python(16),
                       pycpp::to_python(1),
                       pycpp::to_python(1),
                       pycpp::to_python("rm.dat")});

  test("save_tensor", {pycpp::to_python(1),
                       pycpp::to_python(16),
                       pycpp::to_python(1),
                       pycpp::to_python(1),
                       pycpp::to_python("rv.dat")});

  test("save_tensor", {pycpp::to_python(2), // batch
                       pycpp::to_python(3), // in_filters
                       pycpp::to_python(224), // change back to 224
                       pycpp::to_python(224), // change back to 224
                       pycpp::to_python("img.dat")});

  test("save_tensor", {pycpp::to_python(1),  // no batch
                       pycpp::to_python(1),  // no filters
                       pycpp::to_python(3),  // output size
                       pycpp::to_python(55*56*16),  // input size
                       pycpp::to_python("lin_weight.dat")});

  test("save_tensor", {pycpp::to_python(1),  // no batch
                       pycpp::to_python(1),  // no filters
                       pycpp::to_python(3),  // output size
                       pycpp::to_python(1),  // vector
                       pycpp::to_python("lin_bias.dat")});

  PyObject *i = test("load_tensor", {pycpp::to_python("img.dat")});

  af::timer::start();
  PyObject *pto = test("test_conv", {pycpp::to_python("filts.dat"),
                                     pycpp::to_python("bias.dat"), pycpp::to_python("img.dat"),
                                     pycpp::to_python("lin_weight.dat"), pycpp::to_python("lin_bias.dat"),
                                     pycpp::to_python("gamma.dat"), pycpp::to_python("beta.dat"),
                                     pycpp::to_python("rm.dat"),  pycpp::to_python("rv.dat")});
  std::cout << "pytorch forward took (s): " << af::timer::stop() << std::endl;

  // Initialize the engine (sets up the backend)
  pytorch::inference_engine engine (0, AF_BACKEND_OPENCL, false);

  // Load up the image and target
  auto image = pytorch::from_numpy((PyArrayObject *)i, 4, {2, 3, 224, 224});
  auto pytorch_out = pytorch::from_numpy((PyArrayObject *)pto, 4, {2, 3, 1, 1});
  // need to reorder if it's coming out of a linear layer
  pytorch_out = af::array(pytorch_out, 3, 1, 1, 2);  // Get the output from pytorch - this is the result we
                                                       // want to replicate

  pytorch::conv_params_t params = {3, 1, 1, 1, 0, 0};  // filter_x, filter_y, stride_x, stride_y, pad_x, pad_y, has_bias
  pytorch::pooling_params_t poolparams = {2, 2, 2, 2, 0, 0};

  // Set up the layers of our network
//  pytorch::Conv2d conv(params, "filts.dat", {1, 3, 3, 5}, "bias.dat", {1, 1, 1, 1});
//  pytorch::Linear lin("lin_weight.dat", {1, 1, 3, 8}, "lin_bias.dat", {1, 1, 3, 1});
//  pytorch::Hardtanh hardtanh(-1, 1);

  // Can set up layers like above (commented) or this way, both have the same effect.
  engine.add_layer(new pytorch::Conv2d(params, "filts.dat", {16, 3, 3, 1}, false, "bias.dat", {1, 16, 1, 1}));
  engine.add_layer(new pytorch::BatchNorm2d("gamma.dat", {1, 16, 1, 1},
                                            "beta.dat", {1, 16, 1, 1},
                                            "rm.dat", {1, 16, 1, 1},
                                            "rv.dat", {1, 16, 1, 1}, 1e-5));
  engine.add_layer(new pytorch::Tanh);
  engine.add_layer(new pytorch::MaxPool2d(poolparams));
  engine.add_layer(new pytorch::Sigmoid);
  engine.add_layer(new pytorch::MaxPool2d(poolparams));
  engine.add_layer(new pytorch::Hardtanh(-0.1f, 0.1f));
  engine.add_layer(new pytorch::Linear("lin_weight.dat", {1, 1, 3, 55*56*16}, false, "lin_bias.dat", {1, 1, 3, 1}));
  engine.add_layer(new pytorch::ReLU);
//  engine.add_layer(new pytorch::Softmax);  // also tends to end up being unstable
  af::timer::start();
  auto output = engine.forward(image);
  std::cout << "forward took (s): " << af::timer::stop() << std::endl;

  af_print((pytorch_out - output) / af::max(af::constant(1.f, output.dims()), output));
                                             // normalize error by the size of the number
                                             // (some small numerical error for huge numbers)
//  af_print(pytorch_out);
//  af_print(output);

  return 0;
}