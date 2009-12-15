// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

void f();

// FIXME: would like to refer to the first function parameter in these test,
// but that won't work (yet).

// Test typeof(expr) canonicalization
template<typename T, T N>
void f0(T x, decltype(f(N)) y) { } // expected-note{{previous}}

template<typename T, T N>
void f0(T x, decltype((f)(N)) y) { }

template<typename U, U M>
void f0(U u, decltype(f(M))) { } // expected-error{{redefinition}}
