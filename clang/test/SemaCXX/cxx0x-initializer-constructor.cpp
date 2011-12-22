// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

namespace objects {

  struct X1 { X1(int); };
  struct X2 { explicit X2(int); }; // expected-note 2 {{candidate constructor}}

  template <int N>
  struct A {
    A() { static_assert(N == 0, ""); }
    A(int, double) { static_assert(N == 1, ""); }
  };

  template <int N>
  struct E {
    E(int, int) { static_assert(N == 0, ""); }
    E(X1, int) { static_assert(N == 1, ""); }
  };

  void overload_resolution() {
    { A<0> a{}; }
    { A<0> a = {}; }
    { A<1> a{1, 1.0}; }
    { A<1> a = {1, 1.0}; }

    { E<0> e{1, 2}; }
  }

  void explicit_implicit() {
    { X1 x{0}; }
    { X1 x = {0}; }
    { X2 x{0}; }
    { X2 x = {0}; } // expected-error {{no matching constructor}}
  }

  struct C {
    C();
    C(int, double);
    C(int, int);

    int operator[](C);
  };

  C function_call() {
    void takes_C(C);
    takes_C({1, 1.0});

    //C c;
    //c[{1, 1.0}]; needs overloading

    return {1, 1.0};
  }

  void inline_init() {
    (void) C{1, 1.0};
    (void) new C{1, 1.0};
  }

  struct B {
    B(C, int, C);
  };

  void nested_init() {
    //B b{{1, 1.0}, 2, {3, 4}}; needs overloading
  }
}
