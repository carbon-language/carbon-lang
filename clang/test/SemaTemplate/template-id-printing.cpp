// RUN: clang-cc -fsyntax-only -ast-print %s | FileCheck %s
namespace N {
  template<typename T, typename U> void f(U);
  template<int> void f();
}

void g() {
  // CHECK: N::f<int>(3.14
  N::f<int>(3.14);
  
  // CHECK: N::f<double>
  void (*fp)(int) = N::f<double>;
}
