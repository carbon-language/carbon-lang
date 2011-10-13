// RUN: %clang_cc1 %s -std=c++11 -emit-llvm -o - | FileCheck %s

bool b();
struct S {
  int n = b() ? S().n + 1 : 0;
};

S s;

// CHECK: define {{.*}} @_ZN1SC2Ev(
// CHECK-NOT }
// CHECK: call {{.*}} @_Z1bv()
// CHECK-NOT }
// CHECK: call {{.*}} @_ZN1SC1Ev(
