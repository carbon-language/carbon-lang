// RUN: clang-cc -fsyntax-only -verify %s

template<template<typename T> class X> struct A; // expected-note 2{{previous template template parameter is here}}

template<template<typename T, int I> class X> struct B; // expected-note{{previous template template parameter is here}}

template<template<int I> class X> struct C;  // expected-note{{previous non-type template parameter with type 'int' is here}}

template<class> struct X; // expected-note{{too few template parameters in template template argument}}
template<int N> struct Y; // expected-note{{template parameter has a different kind in template argument}}
template<long N> struct Ylong; // expected-note{{template non-type parameter has a different type 'long' in template argument}}

namespace N {
  template<class> struct Z;
}
template<class, class> struct TooMany; // expected-note{{too many template parameters in template template argument}}


A<X> *a1; 
A<N::Z> *a2;
A< ::N::Z> *a3;

A<Y> *a4; // expected-error{{template template argument has different template parameters than its corresponding template template parameter}}
A<TooMany> *a5; // expected-error{{template template argument has different template parameters than its corresponding template template parameter}}
B<X> *a6; // expected-error{{template template argument has different template parameters than its corresponding template template parameter}}
C<Y> *a7;
C<Ylong> *a8; // expected-error{{template template argument has different template parameters than its corresponding template template parameter}}

template<typename T> void f(int);

A<f> *a9; // expected-error{{must be a class template}}

// FIXME: The code below is ill-formed, because of the evil digraph '<:'. 
// We should provide a much better error message than we currently do.
// A<::N::Z> *a10;
