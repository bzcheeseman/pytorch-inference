cmake_minimum_required(VERSION 3.5)
project(pytorch_inference)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

# Documentation ######################################################################
find_package(Doxygen)

if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile @ONLY)
    add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
            SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs/
            COMMENT "Generating API documentation with Doxygen" VERBATIM
            )
endif(DOXYGEN_FOUND)

# Documentation ######################################################################

# From Eyescale/CMake - brilliant solution ############################################
if (PYTHON_EXECUTABLE)
    # Find out the include path
    execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" -c
            "from __future__ import print_function\ntry: import numpy; print(numpy.get_include(), end='')\nexcept:pass\n"
            OUTPUT_VARIABLE __numpy_path)
    # And the version
    execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" -c
            "from __future__ import print_function\ntry: import numpy; print(numpy.__version__, end='')\nexcept:pass\n"
            OUTPUT_VARIABLE __numpy_version)
elseif(__numpy_out)
    message(STATUS "Python executable not found.")
endif(PYTHON_EXECUTABLE)

find_path(PYTHON_NUMPY_INCLUDE_DIR numpy/arrayobject.h
        HINTS "${__numpy_path}" "${PYTHON_INCLUDE_PATH}" NO_DEFAULT_PATH)

include_directories(${PYTHON_INCLUDE_PATH} ${PYTHON_NUMPY_INCLUDE_DIR})

# From Eyescale/CMake - brilliant solution ############################################

# Required for Py_Cpp - need to set the correct executable and source dir
add_definitions(-DPYCPP_PY_HOME="${CMAKE_CURRENT_SOURCE_DIR}/scripts")
add_definitions(-DPYCPP_WHICH_PYTHON="${PYTHON_EXECUTABLE}")

# ArrayFire is the OpenCL backend, from ArrayFire #####################################
find_package(ArrayFire REQUIRED)
include_directories(${ArrayFire_INCLUDE_DIRS})

# If you intend to use OpenCL, you need to find it
find_package(OpenCL)
set(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${OpenCL_LIBRARIES})

# If cuda works, then uncomment this (and make sure you installed arrayfire with CUDA!)
#find_package(CUDA)
#find_package(NVVM) # this FIND script can be found in the ArrayFire CMake example repository
#set(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CUDA_LIBRARIES} ${NVVM_LIB})

# ArrayFire is the OpenCL backend, from ArrayFire #####################################

set(SOURCE_FILES
        main.cpp
        include/inference_engine.hpp
        include/utils.hpp
        include/layers.hpp
        include/modules/Layer.hpp
        include/modules/Convolution.hpp
        include/modules/Pooling.hpp
        include/modules/Normalization.hpp
        include/modules/Linear.hpp
        include/modules/Branch.hpp
        include/modules/Slice.hpp
        include/modules/Concatenate.hpp
        include/modules/Activations.hpp
        include/modules/Activations_Impl.hpp
        include/modules/Branch_Impl.hpp
        include/modules/Concatenate_Impl.hpp
        include/modules/Convolution_Impl.hpp
        include/modules/Pooling_Impl.hpp
        include/modules/Linear_Impl.hpp
        include/modules/Normalization_Impl.hpp
        include/modules/Slice_Impl.hpp)

set(TEST_SOURCE_FILES
        test/utils.hpp)

set(TESTS_BUILT)

MACRO(add_test name)
    set(TESTS_BUILT test_${name} ${TESTS_BUILT})
    add_executable(test_${name} ${TEST_SOURCE_FILES} test/test_${name}.cpp)
    target_link_libraries(test_${name} ${PYTHON_LIBRARY} ${EXTRA_LIBS} ${ArrayFire_LIBRARIES})
ENDMACRO()

add_test(activation)
add_test(branch)
add_test(concat)
add_test(convolution)
add_test(linear)
add_test(normalization)
add_test(pooling)
add_test(slice)

MACRO(test_all)
    add_custom_target(test_all
            COMMAND "../test/run_tests.sh"
            COMMENT "Building all unit tests" VERBATIM
            )
ENDMACRO()

test_all()

add_executable(pytorch_inference ${SOURCE_FILES})
target_link_libraries(pytorch_inference  ${PYTHON_LIBRARY} ${EXTRA_LIBS} ${ArrayFire_LIBRARIES})
