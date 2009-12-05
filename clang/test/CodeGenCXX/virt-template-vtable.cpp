// RUN: clang-cc %s -emit-llvm -o - | FileCheck %s

template<class T> class A {
  A() {}
  virtual void a() {}
};
class B : A<int> {
  B();
};
B::B() {}

// CHECK: @_ZTV1AIiE = weak_odr constant
