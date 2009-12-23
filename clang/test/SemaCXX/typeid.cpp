// RUN: %clang_cc1 -fsyntax-only -verify %s

void f()
{
  (void)typeid(int); // expected-error {{error: you need to include <typeinfo> before using the 'typeid' operator}}
}

// FIXME: This should really include <typeinfo>, but we don't have that yet.
namespace std {
  class type_info;
}

void g()
{
  (void)typeid(int);
}

struct X; // expected-note 3{{forward declaration}}

void g1(X &x) {
  (void)typeid(X); // expected-error{{'typeid' of incomplete type 'struct X'}}
  (void)typeid(X&); // expected-error{{'typeid' of incomplete type 'struct X'}}
  (void)typeid(x); // expected-error{{'typeid' of incomplete type 'struct X'}}
}
