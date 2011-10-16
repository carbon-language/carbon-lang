// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

namespace aggregate {
  // Direct list initialization does NOT allow braces to be elided!
  struct S {
    int ar[2];
    struct T {
      int i1;
      int i2;
    } t;
    struct U {
      int i1;
    } u[2];
    struct V {
      int var[2];
    } v;
  };

  void test() {
    S s1 = { 1, 2, 3 ,4, 5, 6, 7, 8 }; // no-error
    S s2{ {1, 2}, {3, 4}, { {5}, {6} }, { {7, 8} } }; // completely braced
    S s3{ 1, 2, 3, 4, 5, 6 }; // expected-error 5 {{cannot omit braces}}
    S s4{ {1, 2}, {3, 4}, {5, 6}, { {7, 8} } }; // expected-error 2 {{cannot omit braces}}
    S s5{ {1, 2}, {3, 4}, { {5}, {6} }, {7, 8} }; // expected-error {{cannot omit braces}}
  }
}
