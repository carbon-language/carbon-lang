// RUN: clang-cc -fsyntax-only -verify %s

template<typename T, int N = 2> struct X; // expected-note{{template is declared here}}

X<int, 1> *x1;
X<int> *x2;

X<> *x3; // expected-error{{too few template arguments for class template 'X'}}

template<typename U = float, int M> struct X;

X<> *x4;
