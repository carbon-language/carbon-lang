// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct S {
  S *p = this; // ok
  decltype(this) q; // expected-error {{invalid use of 'this' outside of a nonstatic member function}} \
                       expected-error {{C++ requires a type specifier for all declarations}}

  int arr[sizeof(this)]; // expected-error {{invalid use of 'this' outside of a nonstatic member function}}
  int sz = sizeof(this); // ok
};
