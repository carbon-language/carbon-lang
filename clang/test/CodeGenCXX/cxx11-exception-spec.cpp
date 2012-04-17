// RUN: %clang_cc1 -std=c++11 -verify -emit-llvm %s -o - | FileCheck %s

template<typename T> void f() noexcept(sizeof(T) == 4);

void g() {
  // CHECK: declare void @_Z1fIiEvv() nounwind
  f<int>();
  // CHECK: declare void @_Z1fIA2_iEvv()
  f<int[2]>();
  // CHECK: declare void @_Z1fIfEvv() nounwind
  void (*f1)() = &f<float>;
  // CHECK: declare void @_Z1fIdEvv()
  void (*f2)() = &f<double>;
  // CHECK: declare void @_Z1fIA4_cEvv() nounwind
  (void)&f<char[4]>;
  // CHECK: declare void @_Z1fIcEvv()
  (void)&f<char>;
}
