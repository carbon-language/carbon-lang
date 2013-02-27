// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace PR15360 {
  template<typename R, typename U, R F>
  U f() { return &F; } // expected-error{{cannot take the address of an rvalue of type 'int (*)(int)'}} expected-error{{cannot take the address of an rvalue of type 'int *'}}
  void test() {
    f<int(int), int(*)(int), nullptr>(); // expected-note{{in instantiation of}}
    f<int[3], int*, nullptr>(); // expected-note{{in instantiation of}}
  }
}
