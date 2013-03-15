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
    // CHECK: no viable conversion from 'A<[...], (default) x + 1 aka 1>' to 'A<[...], 0>'
    A<0, 0> N = M;
   // CHECK: no viable conversion from 'A<0, [...]>' to 'A<1, [...]>'
    A<1, 1> O = M;
  }
}

namespace default_args {
  template <int x, int y = 1+1, int z = 2>
  class A {};

  void foo(A<0> &M) {
    // CHECK: no viable conversion from 'A<[...], (default) 1 + 1 aka 2, (default) 2>' to 'A<[...], 0, 0>'
    A<0, 0, 0> N = M;

    // CHECK: no viable conversion from 'A<[2 * ...], (default) 2>' to 'A<[2 * ...], 0>'
    A<0, 2, 0> N2 = M;
  }

}
