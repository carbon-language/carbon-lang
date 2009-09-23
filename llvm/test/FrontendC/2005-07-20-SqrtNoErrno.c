// RUN: %llvmgcc %s -S -o - -fno-math-errno | grep llvm.sqrt
// llvm.sqrt has undefined behavior on negative inputs, so it is
// inappropriate to translate C/C++ sqrt to this.
// XFAIL: *
#include <math.h>

float foo(float X) {
  // Check that this compiles to llvm.sqrt when errno is ignored.
  return sqrtf(X);
}
