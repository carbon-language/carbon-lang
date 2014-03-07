// RUN: %clang_cc1 -std=c++1y -fsyntax-only -verify %s

// -- The argument list of the specialization shall not be identical
//    to the implicit argument list of the primary template.

template<typename T, int N, template<typename> class X> int v1;
template<typename T, int N, template<typename> class X> int v1<T, N, X>;
// expected-error@-1{{variable template partial specialization does not specialize any template argument; to define the primary template, remove the template argument list}}

template<typename...T> int v2;
template<typename...T> int v2<T...>;
// expected-error@-1{{variable template partial specialization does not specialize any template argument; to define the primary template, remove the template argument list}}

template<int...N> int v3;
template<int...N> int v3<N...>;
// expected-error@-1{{variable template partial specialization does not specialize any template argument; to define the primary template, remove the template argument list}}

template<template<typename> class...X> int v4;
template<template<typename> class...X> int v4<X...>;
// expected-error@-1{{variable template partial specialization does not specialize any template argument; to define the primary template, remove the template argument list}}

template<typename Outer> struct X {
  template<typename Inner> static int y;
  template<typename Inner> static int y<Outer>; // expected-warning {{cannot be deduced}} expected-note {{'Inner'}}
  template<typename Inner> static int y<Inner>; // expected-error {{does not specialize}}
};
template<typename Outer> template<typename Inner> int X<Outer>::y<Outer>; // expected-warning {{cannot be deduced}} expected-note {{'Inner'}}
template<typename Outer> template<typename Inner> int X<Outer>::y<Inner>; // expected-error {{does not specialize}}

// FIXME: Merging this with the above class causes an assertion failure when
// instantiating one of the bogus partial specializations.
template<typename Outer> struct Y {
  template<typename Inner> static int y;
};
template<> template<typename Inner> int Y<int>::y<Inner>; // expected-error {{does not specialize}}
