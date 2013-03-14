// RUN: %clang_cc1 -fsyntax-only %s -std=c++98 2>&1 | FileCheck %s

namespace PR14342 {
  template<typename T, char a> struct X {};
  X<int, 1> x = X<long, 257>();
  // CHECK: error: no viable conversion from 'X<long, [...]>' to 'X<int, [...]>'
}

namespace PR15513 {
  template <int x, int y = x+1>
  class A {};

  void foo(A<0> &M) {
    // CHECK: no viable conversion from 'A<[...], (default) x + 1>' to 'A<[...], 0>'
    A<0, 0> N = M;
  }
}
