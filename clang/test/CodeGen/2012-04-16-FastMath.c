// RUN: %clang_cc1 %s -ffast-math -emit-llvm -o - | FileCheck %s

double add(double x, double y) {
// CHECK: @add
  return x + y;
// CHECK: fadd double %{{.}}, %{{.}}, !fpmath !0
}
// CHECK: !0 = metadata !{metadata !"fast"}
