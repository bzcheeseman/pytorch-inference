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
  test("save_tensor", {pycpp::to_python(32), // out_filters
                       pycpp::to_python(3), // in_filters
                       pycpp::to_python(3),
                       pycpp::to_python(1),
                       pycpp::to_python("filts.dat")});

  test("save_tensor", {pycpp::to_python(1),
                       pycpp::to_python(32), // out_filters
                       pycpp::to_python(1),
                       pycpp::to_python(1),
                       pycpp::to_python("bias.dat")});

  test("save_tensor", {pycpp::to_python(10), // batch
                       pycpp::to_python(3), // in_filters
                       pycpp::to_python(224),
                       pycpp::to_python(224),
                       pycpp::to_python("img.dat")});

  test("save_tensor", {pycpp::to_python(1),  // no batch
                       pycpp::to_python(1),  // no filters
                       pycpp::to_python(30),  // output size
                       pycpp::to_python(222*224*32),  // input size
                       pycpp::to_python("lin_weight.dat")});

  test("save_tensor", {pycpp::to_python(1),  // no batch
                       pycpp::to_python(1),  // no filters
                       pycpp::to_python(30),  // output size
                       pycpp::to_python(1),  // vector
                       pycpp::to_python("lin_bias.dat")});

  PyObject *i = test("load_tensor", {pycpp::to_python("img.dat")});
  PyObject *fs = test("load_tensor", {pycpp::to_python("filts.dat")});

  af::timer::start();
  PyObject *pto = test("test_conv", {pycpp::to_python("filts.dat"),
                                        pycpp::to_python("bias.dat"), pycpp::to_python("img.dat"),
                                        pycpp::to_python("lin_weight.dat"), pycpp::to_python("lin_bias.dat")});
  std::cout << "pytorch forward took (s): " << af::timer::stop() << std::endl;

  // dimension 2 in pytorch goes to dimension 0 in arrayfire
//  PyObject *cato = test("test_concat", {pycpp::to_python("filts.dat"), pycpp::to_python(2)});

  // Initialize the engine (sets up the backend)
  pytorch::inference_engine engine;

//  auto f = pytorch::from_numpy((PyArrayObject *)fs, 4, {32, 3, 3, 1});
//  auto pytorch_cato = pytorch::from_numpy((PyArrayObject *)cato, 4, {32, 3, 6, 1});
//  auto cat_test = pytorch::cat2(f, f, 0); // gotta get the layer figured out

  // Load up the image and target
  auto image = pytorch::from_numpy((PyArrayObject *)i, 4, {10, 3, 224, 224});
  auto pytorch_out = pytorch::from_numpy((PyArrayObject *)pto, 4, {10, 30, 1, 1}); // need to reorder - not 4-dim originally
  pytorch_out = af::reorder(pytorch_out, 2, 1, 0, 3);  // Get the output from pytorch - this is the result we
                                                       // want to replicate

  pytorch::conv_params_t params = {3, 1, 1, 1, 0, 0};  // filter_x, filter_y, stride_x, stride_y, pad_x, pad_y

  // Set up the layers of our network
//  pytorch::Conv2d conv(params, "filts.dat", {1, 3, 3, 5}, "bias.dat", {1, 1, 1, 1});
//  pytorch::Linear lin("lin_weight.dat", {1, 1, 3, 8}, "lin_bias.dat", {1, 1, 3, 1});
//  pytorch::Hardtanh hardtanh(-1, 1);

  // Can set up layers like above (commented) or this way, both have the same effect.
  engine.add_layer(new pytorch::Conv2d(params, "filts.dat", {32, 3, 3, 1}, "bias.dat", {1, 32, 1, 1}));
  engine.add_layer(new pytorch::Linear("lin_weight.dat", {1, 1, 30, 222*224*32}, "lin_bias.dat", {1, 1, 30, 1}));
  engine.add_layer(new pytorch::Hardtanh(-1, 1));
  af::timer::start();
  auto output = engine.forward(image);
  std::cout << "forward took (s): " << af::timer::stop() << std::endl;

  af_print(pytorch_out - output);

  return 0;
}