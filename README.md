# nanobind_package_example

demo python package using [nanobind](https://github.com/wjakob/nanobind) and [scikit-build-core](https://github.com/scikit-build/scikit-build-core)

## Install and run

In a virtual environement (`Conda`), git clone the repository and run the cmd `pip install -e . -v` to build and install the package.

## Structure

The repo is a simple playground code for a python package `mypackage` which hold a module `tissue` that is entirely build in C++ and made visible using nanobind.
The build is managed by scikit-build-core and CMake.