// RUN: %clang_cc1 -ffinite-math-only -emit-llvm -o - %s | FileCheck %s
float f0, f1, f2;

void foo(void) {
  // CHECK: define void @foo()

  // CHECK: fadd nnan ninf
  f0 = f1 + f2;

  // CHECK: ret
}
