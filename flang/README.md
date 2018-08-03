<!--
Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
-->

# f18

F18 is a front-end for Fortran.
It is intended to replace the existing front-end in the Flang compiler.

Flang is a Fortran compiler targeting LLVM.

Visit the Flang wiki for more information about Flang:

https://github.com/flang-compiler/flang/wiki

Read more about f18 in the documentation directory.

## Building F18

### Selection of the C++ compiler

F18 is written in C++17.

The code has been compiled and tested with
GCC versions 7.2.0, 7.3.0, 8.1.0, and 8.2.0.
The code has been compiled and tested with clang 6.0
using either GCC 7.3.0 or 8.1.0 headers;
however, the headers needed small patches.

To build and install f18, there are several options
for specifying the C++ compiler.
You can have the proper C++ compiler on your path,
or you can set the environment variable CXX,
or you can define the variable GCC on the cmake command line.

By default,
cmake will search for g++ on your PATH.
The g++ version must be 7.2 or greater in order to build f18.

Or, if you export CXX,
cmake will use the variable CXX to find the C++ compiler.
CXX should include the full path to the compiler
or a name that will be found on your PATH,
e.g. g++-7.2, assuming g++-7.2 is on your PATH.
```
export CXX=g++-7.2
```

Or, you can reference the GCC installation directory directly.
The CMakeList.txt file
uses the variable GCC
as the path to the bin directory
containing the C++ compiler.
GCC can be defined on the cmake command line
where `<GCC_DIRECTORY>` is the path to a GCC installation with bin, lib, etc:
```
cmake -DGCC=<GCC_DIRECTORY>
```

To use f18 after it is built,
the environment variables PATH and LD_LIBRARY_PATH
must be set to use GCC and its associated libraries.

### LLVM and Clang dependency

F18 uses components from version 6.0 of LLVM and clang
(even when f18 is not compiled with clang).

The instructions to build LLVM and clang can be found at
https://clang.llvm.org/get_started.html.

The f18 CMakeList.txt file uses
the environment variable `Clang_DIR` to find the installed components.

To get the correct LLVM and clang libraries included in your f18 build,
set the environment variable
`Clang_DIR`
to the `lib/cmake/clang` directory in the clang install directory.

### Installation Directory

To specify a custom install location,
add
`-DCMAKE_INSTALL_PREFIX=<INSTALL_PREFIX>`
to the cmake command
where `<INSTALL_PREFIX>`
is the path where f18 should be installed.

### Build Types

To create a debug build,
add
`-DCMAKE_BUILD_TYPE=Debug`
to the cmake command.
Debug builds execute slowly.

To create a release build,
add
`-DCMAKE_BUILD_TYPE=Release`
to the cmake command.
Release builds execute quickly.

### Get the Source Code

```
cd where/you/want/the/source
git clone https://github.com/flang-compiler/f18.git
```
### Build F18
```
cd where/you/want/to/build
export Clang_DIR=<CLANG_CMAKE_DIRECTORY>
cmake <your custom options> where/you/put/the/source/f18
make
```
