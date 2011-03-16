// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x

void f(); // expected-note{{candidate function}}
void f(int); // expected-note{{candidate function}}
decltype(f) a; // expected-error{{cannot resolve overloaded function 'f' from context}}

template<typename T> struct S {
  decltype(T::f) * f; // expected-error{{cannot resolve overloaded function 'f' from context}}
};

struct K { 
  void f();  // expected-note{{candidate function}}
  void f(int); // expected-note{{candidate function}}
};
S<K> b; // expected-note{{in instantiation of template class 'S<K>' requested here}}
