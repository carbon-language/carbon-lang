// RUN: clang-tidy -checks='-*,google-explicit-constructor' %s -- | FileCheck %s

template<typename T>
struct A { A(T); };
// CHECK: :[[@LINE-1]]:12: warning: single-argument constructors must be explicit [google-explicit-constructor]
// CHECK-NOT: warning:


void f() {
  A<int> a(0);
  A<double> b(0);
}
