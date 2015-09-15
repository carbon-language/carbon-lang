CMake Caches
============

This directory contains CMake cache scripts that pre-populate the CMakeCache in
a build directory with commonly used settings.

The first two cache files in the directory are used by Apple to build the clang
distribution packaged with Xcode. You can use the caches with the following
CMake invocation:

cmake -G <build system>
  -C <path to llvm>/tools/clang/cmake/caches/Apple-stage1.cmake
  -DCMAKE_BUILD_TYPE=Release
  [-DCMAKE_INSTALL_PREFIX=<install path>]
  <path to llvm>

Building the `bootstrap` target from this generation will build clang, and
`bootstrap-install` will install it.
