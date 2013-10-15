#!/bin/bash
set -u
set -e

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -d "$ROOT/build" ]; then
  cd $ROOT/build
else
  mkdir -p $ROOT/build
  cd $ROOT/build
  CC=clang CXX=clang++ cmake -G Ninja -DLLVM_ENABLE_WERROR=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON $ROOT/../../../..
fi
ninja
ninja check-sanitizer
ninja check-tsan
ninja check-asan
ninja check-msan
ninja check-lsan
