// RUN: %clang_cc1 %s -cl-single-precision-constant -emit-llvm -o - | FileCheck %s

float fn(float f) {
  // CHECK: fmul float
  // CHECK: fadd float
  return f*2. + 1.;
}
