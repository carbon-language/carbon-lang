// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

void f(); // expected-note{{possible target for call}}
void f(int); // expected-note{{possible target for call}}
decltype(f) a;  // expected-error{{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}} expected-error {{variable has incomplete type 'decltype(f())' (aka 'void')}}

template<typename T> struct S {
  decltype(T::f) * f; // expected-error{{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}} expected-error {{call to non-static member function without an object argument}}
};

struct K { 
  void f();  // expected-note{{possible target for call}}
  void f(int); // expected-note{{possible target for call}}
};
S<K> b; // expected-note{{in instantiation of template class 'S<K>' requested here}}
