// RUN: clang-tidy -checks='-*,google-explicit-constructor' %s -- | FileCheck %s

template<typename T>
class A { A(T); };
// CHECK: :[[@LINE-1]]:11: warning: Single-argument constructors must be explicit [google-explicit-constructor]
// CHECK-NOT: warning:


void f() {
  A<int> a;
  A<double> b;
}
