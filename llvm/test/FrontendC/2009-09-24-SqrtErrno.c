// RUN: %llvmgcc %s -S -o - -fmath-errno | FileCheck %s
// llvm.sqrt has undefined behavior on negative inputs, so it is
// inappropriate to translate C/C++ sqrt to this.
#include <math.h>

float foo(float X) {
// CHECK: foo
// CHECK-NOT: readonly
// CHECK: return
  // Check that this is not marked readonly when errno is used.
  return sqrtf(X);
}
