#!/bin/bash
set -u
set -e

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p $ROOT/build
cd $ROOT/build
CC=gcc CXX=g++ cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON $ROOT/../../../..
make -j64
make check-tsan check-sanitizer -j64

