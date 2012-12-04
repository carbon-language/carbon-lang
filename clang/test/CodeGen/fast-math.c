// RUN: %clang_cc1 -ffast-math -emit-llvm -o - %s | FileCheck %s
typedef unsigned cond_t;

volatile float f0, f1, f2;

void foo(void) {
  // CHECK: define void @foo()

  // CHECK: fadd fast
  f0 = f1 + f2;

  // CHECK: ret
}
