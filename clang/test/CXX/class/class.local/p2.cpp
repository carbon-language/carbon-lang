// RUN: clang-cc -fsyntax-only -verify %s -faccess-control

struct A { };

void f() {
  struct B : private A {}; // expected-note{{'private' inheritance specifier here}}
  
  B b;
  
  A *a = &b; // expected-error{{conversion from 'struct B' to inaccessible base class 'struct A'}} \
                expected-error{{incompatible type initializing 'struct B *', expected 'struct A *'}}
}
