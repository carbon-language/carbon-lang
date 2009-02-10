// RUN: clang -fsyntax-only -std=c++98 -verify %s

template<int N> struct A; // expected-note 2{{template parameter is declared here}}

A<0> *a0;

A<int()> *a1; // expected-error{{template argument for non-type template parameter is treated as type 'int (void)'}}

A<int> *a2; // expected-error{{template argument for non-type template parameter must be an expression}}

A<1 >> 2> *a3;

// FIXME: We haven't tried actually checking the expressions yet.
// A<A> *a4; 
