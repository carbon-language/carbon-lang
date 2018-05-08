<!--
Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
-->

# f18

## Selection of the C/C++ compiler

F18 requires a C++17 compiler.  The code has been tested with g++
7.2.0, g++ 7.3.0, g++ 8.1.0, and clang 6.0 (using g++ 7.3.0 headers).

For a proper installation, we assume that the PATH and LD_LIBRARY_PATH environment variables
are properly set to use gcc, g++ and the associated libraries.

cmake will require that the environment variables CC and CXX are properly set (else it would
search for the 'cc' and 'c++' commands which are likely /usr/bin/cc and /usr/bin/c++).
That can be done now or while calling cmake.

    export CC=gcc
    export CXX=g++

## Installation of LLVM & CLANG 6.0

F18 depends on the LLVM & CLANG libraries even when clang is not used as C++ compiler.

If those libraries are not provided by your system, then you may want to follow the
build instructions at https://clang.llvm.org/get_started.html .

## Installation of F18

    ######## Choose the installation directory

    F18_PREFIX=...

    ######## Get Flang sources in $ROOT/f18
    cd $ROOT
    git clone https://github.com/ThePortlandGroup/f18.git

    ######## And build f18 in a dedicated directory
    ######## Reminder: If LLVM & Clang were not installed in a standard
    ########           location, then you may also have to specify it via the
    ########           CMAKE_MODULE_PATH or CMAKE_PREFIX_PATH variables.
    mkdir $ROOT/f18-build
    cd $ROOT/f18-build
    CC=gcc CXX=g++ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$F18_PREFIX $ROOT/f18
    make -j 8
    make install
