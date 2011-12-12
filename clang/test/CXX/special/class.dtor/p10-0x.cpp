// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// PR10127/N3031
struct A { ~A(); };
struct B {};
template<typename T>
void b(const T *x, const A *y) {
  // FIXME: this parses as a pseudo destructor call which doesn't have decltype support yet
  x->~decltype(T())(); // expected-error{{expected a class name after '~' to name a destructor}}

  y->~decltype(*y)(); // expected-error{{destructor type 'decltype(*y)' (aka 'const A &') in object destruction expression does not match the type 'const A' of the object being destroyed}}
  y->~decltype(T())(); // expected-error{{destructor type 'decltype(T())' in object destruction expression does not match the type 'const A' of the object being destroyed}}
  y->~decltype(A())();
}
template void b(const int*, const A*);
template void b(const A*,const A*);
void a(const A *x) {
  x->~decltype(A())();
  x->~decltype(*x)(); // expected-error{{destructor type 'decltype(*x)' (aka 'const A &') in object destruction expression does not match the type 'const A' of the object being destroyed}}
  x->~decltype()(); // expected-error{{expected expression}}
  x->~decltype(B())(); // expected-error{{destructor type 'decltype(B())' (aka 'B') in object destruction expression does not match the type 'const A' of the object being destroyed}}
  x->~decltype(x)(); // expected-error{{destructor type 'decltype(x)' (aka 'const A *') in object destruction expression does not match the type 'const A' of the object being destroyed}}
  // this last one could be better, mentioning that the nested-name-specifier could be removed or a type name after the ~
  x->::A::~decltype(*x)(); // expected-error{{expected a class name after '~' to name a destructor}}
  y->~decltype(A())(); // expected-error{{use of undeclared identifier 'y'}}
}
