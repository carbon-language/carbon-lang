// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

template<typename T>
struct X {
  void f() {}
};

template inline void X<int>::f(); // expected-error{{explicit instantiation cannot be 'inline'}}

template<typename T>
struct Y {
  constexpr int f() { return 0; }
};

template constexpr int Y<int>::f(); // expected-error{{explicit instantiation cannot be 'constexpr'}}
