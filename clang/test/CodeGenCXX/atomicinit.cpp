// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-apple-darwin9 | FileCheck %s
struct A {
  _Atomic(int) i;
  A(int j);
  void v(int j);
};
// Storing to atomic values should be atomic
// CHECK: store atomic i32
void A::v(int j) { i = j; }
// Initialising atomic values should not be atomic
// CHECK-NOT: store atomic 
A::A(int j) : i(j) {}
