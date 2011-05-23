// RUN: %clang_cc1 -std=c++0x -emit-llvm -o - %s | FileCheck %s

template <typename T>
struct X {
  X();
};

// CHECK: define void @_ZN1XIbEC1Ev
// CHECK: define void @_ZN1XIbEC2Ev
template <> X<bool>::X() = default;

// CHECK: define weak_odr void @_ZN1XIiEC1Ev
// CHECK: define weak_odr void @_ZN1XIiEC2Ev
template <typename T> X<T>::X() = default;
template X<int>::X();

// CHECK: define linkonce_odr void @_ZN1XIcEC1Ev
// CHECK: define linkonce_odr void @_ZN1XIcEC2Ev
X<char> x;
