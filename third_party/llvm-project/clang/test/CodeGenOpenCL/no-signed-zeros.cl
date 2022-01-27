// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s -check-prefix=NORMAL
// RUN: %clang_cc1 %s -emit-llvm -cl-no-signed-zeros -o - | FileCheck %s -check-prefix=NO-SIGNED-ZEROS

float signedzeros(float a) {
  return a;
}

// CHECK: attributes
// NORMAL-NOT: "no-signed-zeros-fp-math"
// NO-SIGNED-ZEROS: "no-signed-zeros-fp-math"="true"
