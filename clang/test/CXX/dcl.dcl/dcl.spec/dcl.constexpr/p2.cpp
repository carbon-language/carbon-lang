// RUN: %clang_cc1 -std=c++0x -emit-llvm %s -o - | FileCheck %s

// constexpr functions and constexpr constructors are implicitly inline.
struct S {
  constexpr S(int n);
  int n;
};

constexpr S::S(int n) : n(n) {}

constexpr S f(S s) {
  return s.n * 2;
}

// CHECK: define linkonce_odr {{.*}} @_Z1f1S(
// CHECK: define linkonce_odr {{.*}} @_ZN1SC1Ei(

int g() {
  return f(42).n;
}
