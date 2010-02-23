// RUN: %clang_cc1 -fsyntax-only -verify %s
struct A {};

enum Foo { F };
typedef Foo Bar;

typedef int Integer;
typedef double Double;

void g();

namespace N {
  typedef Foo Wibble;
}

void f(A* a, Foo *f, int *i, double *d) {
  a->~A();
  a->A::~A();
  
  a->~foo(); // expected-error{{identifier 'foo' in pseudo-destructor expression does not name a type}}
  
  // FIXME: the diagnostic below isn't wonderful
  a->~Bar(); // expected-error{{does not name a type}}
  
  f->~Bar();
  f->~Foo();
  i->~Bar(); // expected-error{{does not match}}
  
  g().~Bar(); // expected-error{{non-scalar}}
  
  f->::~Bar();
  f->N::~Wibble(); // FIXME: technically, Wibble isn't a class-name
  
  f->::~Bar(17, 42); // expected-error{{cannot have any arguments}}

  i->~Integer();
  i->Integer::~Integer();

  i->Integer::~Double(); // expected-error{{the type of object expression ('int') does not match the type being destroyed ('double') in pseudo-destructor expression}}
}

typedef int Integer;

void destroy_without_call(int *ip) {
  ip->~Integer; // expected-error{{called immediately}}
}

// PR5530
namespace N1 {
  class X0 { };
}

void test_X0(N1::X0 &x0) {
  x0.~X0();
}
