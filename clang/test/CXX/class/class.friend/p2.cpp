// RUN: clang-cc -fsyntax-only -verify %s

struct B0;

class A {
  friend class B {}; // expected-error {{cannot define a type in a friend declaration}}
  friend int; // expected-error {{friends can only be classes or functions}}
  friend B0; // expected-error {{must specify 'struct' to befriend}}
  friend class C; // okay
};
