// RUN: clang-cc -emit-llvm %s -o - | FileCheck %s

template <class T> struct A { A(); };
struct B { A<int> x; };
void a() {   
  B b;
}
// CHECK: call void @_ZN1BC1Ev
// CHECK: define linkonce_odr void @_ZN1BC1Ev
// CHECK: call void @_ZN1AIiEC1Ev
