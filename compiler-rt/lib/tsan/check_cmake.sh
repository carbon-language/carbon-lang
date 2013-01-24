#!/bin/bash
set -u
set -e

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p $ROOT/build
cd $ROOT/build
CC=clang CXX=clang++ cmake -DLLVM_ENABLE_WERROR=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON $ROOT/../../../..
make -j64
make check-sanitizer check-tsan check-asan -j64

