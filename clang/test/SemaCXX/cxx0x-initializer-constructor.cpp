// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct one { char c[1]; };
struct two { char c[2]; };

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

    C c;
    c[{1, 1.0}];

    return {1, 1.0};
  }

  void inline_init() {
    (void) C{1, 1.0};
    (void) new C{1, 1.0};
  }

  struct B { // expected-note 2 {{candidate constructor}}
    B(C, int, C); // expected-note {{candidate constructor not viable: cannot convert initializer list argument to 'objects::C'}}
  };

  void nested_init() {
    B b1{{1, 1.0}, 2, {3, 4}};
    B b2{{1, 1.0, 4}, 2, {3, 4}}; // expected-error {{no matching constructor for initialization of 'objects::B'}}
  }

  void overloaded_call() {
    one ov1(B); // expected-note {{not viable: cannot convert initializer list}}
    two ov1(C); // expected-note {{not viable: cannot convert initializer list}}

    static_assert(sizeof(ov1({})) == sizeof(two), "bad overload");
    static_assert(sizeof(ov1({1, 2})) == sizeof(two), "bad overload");
    static_assert(sizeof(ov1({{1, 1.0}, 2, {3, 4}})) == sizeof(one), "bad overload");

    ov1({1}); // expected-error {{no matching function}}
  }
}
