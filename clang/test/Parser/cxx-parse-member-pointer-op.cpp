// RUN: clang-cc -fsyntax-only -pedantic -verify %s

struct C {};

typedef void (C::*pmfc)();

void g(pmfc) {
  C *c;
  c->*pmfc(); // expected-error {{invalid use of pointer to member type after '->*'}}
  C c1;
  c1.*pmfc(); // expected-error {{invalid use of pointer to member type after '.*'}}
}

