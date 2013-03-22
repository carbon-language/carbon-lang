#!/bin/bash

if [ -d "llvm-build" ]; then
    echo "Using existing 'llvm-build' directory..."    
else
    mkdir llvm-build
fi

cd llvm-build

if [ -d "llvm" ]; then
    echo "Using existing 'llvm' directory..."
else
    svn co --revision 176809 http://llvm.org/svn/llvm-project/llvm/trunk llvm
    ( cd llvm/tools ; svn co --revision 176809 http://llvm.org/svn/llvm-project/cfe/trunk clang )
fi

if [ ! -d "build" ]; then
    mkdir build
    cd build
    ../llvm/configure --enable-targets=x86_64,arm --build=x86_64-apple-darwin10 --disable-optimized --disable-assertions --enable-libcpp
    make -j8 DEBUG_SYMBOLS=1
fi



