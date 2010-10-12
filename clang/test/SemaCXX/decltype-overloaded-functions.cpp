// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x

void f();
void f(int);
decltype(f) a; // expected-error{{cannot resolve overloaded function from context}}

template<typename T> struct S {
  decltype(T::f) * f; // expected-error{{cannot resolve overloaded function from context}}
};

struct K { void f(); void f(int); };
S<K> b; // expected-note{{in instantiation of template class 'S<K>' requested here}}
