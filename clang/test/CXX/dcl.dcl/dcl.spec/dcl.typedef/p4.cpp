// RUN: %clang_cc1 -verify %s
// XFAIL: *

struct S {
  typedef struct A {} A; // expected-note {{previous definition is here}}
  typedef struct B B;
  typedef A A; // expected-error {{redefinition of 'A'}}
};

