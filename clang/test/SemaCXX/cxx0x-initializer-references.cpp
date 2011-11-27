// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

namespace reference {
  struct A {
    int i1, i2;
  };

  void single_init() {
    const int &cri1a = {1};
    const int &cri1b{1};

    int i = 1;
    int &ri1a = {i};
    int &ri1b{i};

    int &ri2 = {1}; // expected-error {{cannot bind to an initializer list temporary}}

    A a{1, 2};
    A &ra1a = {a};
    A &ra1b{a};
  }

  void reference_to_aggregate() {
    const A &ra1{1, 2};
    A &ra2{1, 2}; // expected-error {{cannot bind to an initializer list temporary}}

    const int (&arrayRef)[] = {1, 2, 3};
    static_assert(sizeof(arrayRef) == 3 * sizeof(int), "bad array size");
  }

}
