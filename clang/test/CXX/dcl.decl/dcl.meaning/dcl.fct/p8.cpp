// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A { };
A::A (enum { e1 }) {} // expected-error{{can not be defined in a parameter}}
void A::f(enum { e2 }) {} // expected-error{{can not be defined in a parameter}}

enum { e3 } A::g() { } // expected-error{{can not be defined in the result type}} \
// expected-error{{out-of-line definition}}
