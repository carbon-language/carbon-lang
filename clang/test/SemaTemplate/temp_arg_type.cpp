// RUN: clang-cc -fsyntax-only -verify %s
template<typename T> class A; // expected-note 2 {{template parameter is declared here}}

// [temp.arg.type]p1
A<0> *a1; // expected-error{{template argument for template type parameter must be a type}}

A<A> *a2; // expected-error{{template argument for template type parameter must be a type}}

A<int> *a3;
A<int()> *a4; 
A<int(float)> *a5;
A<A<int> > *a6;

// [temp.arg.type]p2
void f() {
  class X { };
  A<X> * a = 0; // expected-error{{template argument uses local type 'class X'}}
}

struct { int x; } Unnamed; // expected-note{{unnamed type used in template argument was declared here}}
A<__typeof__(Unnamed)> *a7; // expected-error{{template argument uses unnamed type}}

// FIXME: [temp.arg.type]p3. The check doesn't really belong here (it
// belongs somewhere in the template instantiation section).
