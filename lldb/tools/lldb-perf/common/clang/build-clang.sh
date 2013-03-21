#!/bin/bash

mkdir llvm-build
cd llvm-build
svn co --revision 176809 http://llvm.org/svn/llvm-project/llvm/trunk llvm
( cd llvm/tools ; svn co --revision 176809 http://llvm.org/svn/llvm-project/cfe/trunk clang )
mkdir build
cd build
../llvm/configure --enable-targets=x86_64,arm --build=x86_64-apple-darwin10 --enable-optimized --disable-assertions
make -j8
