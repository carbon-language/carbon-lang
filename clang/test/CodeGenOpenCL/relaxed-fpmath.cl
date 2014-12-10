// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s -check-prefix=NORMAL
// RUN: %clang_cc1 %s -emit-llvm -cl-fast-relaxed-math -o - | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 %s -emit-llvm -cl-finite-math-only -o - | FileCheck %s -check-prefix=FINITE
// RUN: %clang_cc1 %s -emit-llvm -cl-unsafe-math-optimizations -o - | FileCheck %s -check-prefix=UNSAFE
// RUN: %clang_cc1 %s -emit-llvm -cl-no-signed-zeros -o - | FileCheck %s -check-prefix=NOSZ

typedef __attribute__(( ext_vector_type(4) )) float float4;

float spscalardiv(float a, float b) {
  // CHECK: @spscalardiv(

  // NORMAL: fdiv float    
  // FAST: fdiv fast float
  // FINITE: fdiv nnan ninf float
  // UNSAFE: fdiv nnan float
  // NOSZ: fdiv nsz float
  return a / b;
}
// CHECK: attributes

// NORMAL: "no-infs-fp-math"="false"
// NORMAL: "no-nans-fp-math"="false"
// NORMAL: "unsafe-fp-math"="false"

// FAST: "no-infs-fp-math"="true"
// FAST: "no-nans-fp-math"="true"
// FAST: "unsafe-fp-math"="true"

// FINITE: "no-infs-fp-math"="true"
// FINITE: "no-nans-fp-math"="true"
// FINITE: "unsafe-fp-math"="false"

// UNSAFE: "no-infs-fp-math"="false"
// UNSAFE: "no-nans-fp-math"="true"
// UNSAFE: "unsafe-fp-math"="true"

