// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

struct A { int x; A(int); ~A(); };
A f() { return A(0); }
// CHECK: define void @_Z1fv
// CHECK: call void @_ZN1AC1Ei
// CHECK-NEXT: ret void
