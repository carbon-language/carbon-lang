// RUN: %clang_cc1 -fsyntax-only -verify %s

struct B0;

class A {
  friend class B {}; // expected-error {{cannot define a type in a friend declaration}}
  friend int; // expected-warning {{non-class type 'int' cannot be a friend}}
  friend B0; // expected-warning {{must specify 'struct' to befriend}}
  friend class C; // okay
};
