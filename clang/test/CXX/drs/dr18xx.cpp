// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus < 201103L
// expected-no-diagnostics
#endif

void dr1891() { // dr1891: 3.6
#if __cplusplus >= 201103L
  int n;
  auto a = []{}; // expected-note 2{{candidate}}
  auto b = [=]{ return n; }; // expected-note 2{{candidate}}
  typedef decltype(a) A;
  typedef decltype(b) B;

  static_assert(!__has_trivial_constructor(A), "");
  static_assert(!__has_trivial_constructor(B), "");

  A x; // expected-error {{no matching constructor}}
  B y; // expected-error {{no matching constructor}}
#endif
}
