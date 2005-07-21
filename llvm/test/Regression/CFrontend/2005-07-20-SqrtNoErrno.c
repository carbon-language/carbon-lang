// RUN: %llvmgcc %s -S -o - -fno-math-errno | gccas | llvm-dis | grep llvm.sqrt
#include <math.h>

float foo(float X) {
  // Check that this compiles to llvm.sqrt when errno is ignored.
  return sqrtf(X);
}
