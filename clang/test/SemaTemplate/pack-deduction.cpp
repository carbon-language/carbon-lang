// RUN: %clang_cc1 -std=c++11 -verify %s

template<typename ...T> struct X {};

template<typename T, typename U> struct P {};

namespace Nested {
  template<typename ...T> int f1(X<T, T...>... a); // expected-note +{{conflicting types for parameter 'T'}}
  template<typename ...T> int f2(P<X<T...>, T> ...a); // expected-note +{{conflicting types for parameter 'T'}}

  int a1 = f1(X<int, int, double>(), X<double, int, double>());
  int a2 = f1(X<int, int>());
  int a3 = f1(X<int>(), X<double>()); // expected-error {{no matching}}
  int a4 = f1(X<int, int>(), X<int>()); // expected-error {{no matching}}
  int a5 = f1(X<int>(), X<int, int>()); // expected-error {{no matching}}
  int a6 = f1(X<int, int, int>(), X<int, int, int>(), X<int, int, int, int>()); // expected-error {{no matching}}

  int b1 = f2(P<X<int, double>, int>(), P<X<int, double>, double>());
  int b2 = f2(P<X<int, double>, int>(), P<X<int, double>, double>(), P<X<int, double>, char>()); // expected-error {{no matching}}
}
