<!--
Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
-->

# f18

## Selection of the C/C++ compiler

F18 requires a C++17 compiler. As of today, the code was only tested with g++ 7.2.0 and g++ 7.3.0  

For a proper installation, we assume that the PATH and LD_LIBRARY_PATH environment variables 
are properly set to use gcc, g++ and the associated libraries.   

cmake will require that the environement variables CC and CXX are properly set (else it will 
search for use the 'cc' and 'c++' program which are likely /usr/bin/cc and /usr/bin/c++) that 
can be done now or while calling cmake 

    export CC=gcc
    export CXX=g++

## Installation of LLVM 6.0

    ############ Extract LLVM and Clang from git in current directory. 
    ############ 

    ROOT=$(pwd)
    REL=release_60
   
    # To build LLVM and Clang, we only need the head of the requested branch. 
    # Remove --single-branch --depth=1 if you want access to the whole git history. 
   
    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/llvm.git/       llvm
    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/clang.git/      llvm/tools/clang
    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/openmp.git/     llvm/projects/openmp
    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/libcxx.git/     llvm/projects/libcxx
    git clone --branch $REL --single-branch --depth=1 https://git.llvm.org/git/libcxxabi.git/  llvm/projects/libcxxabi
   
    ###########  Build LLVM & CLANG in $LLVM_PREFIX 

    LLVM_PREFIX=... 
    mkdir $LLVM_PREFIX
    
    mkdir $ROOT/llvm/build
    cd  $ROOT/llvm/build 
    CC=gcc CXX+g++ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$LLVM_PREFIX ..
    make -j 4
    make install
   

## Installation of F18

    ######## Choose the installation directory
   
    F18_PREFIX=...   

    ######## Get Flang sources in $ROOT/f18
    cd $ROOT
    git clone https://github.com/ThePortlandGroup/f18.git

    ######## And build it in a dedicated directory
    ######## Reminder: If LLVM & Clang where not installed in a standard 
    ########           location then you may also have to define
    ########           CMAKE_MODULE_PATH=$LLVM_PREFIX  
    mkdir $ROOT/f18-build
    cd $ROOT/f18-build   
    CC=gcc CXX=g++ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$F18_PREFIX $ROOT/f18
    make -j 4
    make install
