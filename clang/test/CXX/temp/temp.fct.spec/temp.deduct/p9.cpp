// RUN: %clang_cc1 -fsyntax-only -verify %s

template <int> int f(int);  // expected-note 2{{candidate}}
template <signed char> int f(int); // expected-note 2{{candidate}}
int i1 = f<1>(0); // expected-error{{ambiguous}}
int i2 = f<1000>(0); // expected-error{{ambiguous}}
