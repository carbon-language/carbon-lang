// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

void f();
void f(int);
decltype(f) a; // expected-error{{can't determine the declared type of an overloaded function}}

template<typename T> struct S {
  decltype(T::f) * f; // expected-error{{can't determine the declared type of an overloaded function}}
};

struct K { void f(); void f(int); };
S<K> b; // expected-note{{in instantiation of template class 'struct S<struct K>' requested here}}
