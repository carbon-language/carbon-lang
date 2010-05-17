// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct X {
  friend void f(int x) { T* y = x; } // expected-error{{cannot initialize a variable of type 'int *' with an lvalue of type 'int'}}
};

X<int> xi; // expected-note{{in instantiation of member function 'f' requested here}}

