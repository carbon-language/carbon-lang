// RUN: clang-cc -fsyntax-only -verify %s
struct A {};

enum Foo { F };
typedef Foo Bar;

void f(A* a) {
  a->~A();
  a->A::~A();
  
  a->~foo(); // expected-error{{identifier 'foo' in pseudo-destructor expression does not name a type}}
  a->~Bar(); // expected-error{{type 'Bar' (aka 'enum Foo') in pseudo-destructor expression is not a class type}}
}
