// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x

void f();
void f(int);
decltype(f) a; // expected-error{{cannot determine the declared type of an overloaded function}}

template<typename T> struct S {
  decltype(T::f) * f; // expected-error{{cannot determine the declared type of an overloaded function}}
};

struct K { void f(); void f(int); };
S<K> b; // expected-note{{in instantiation of template class 'S<K>' requested here}}
