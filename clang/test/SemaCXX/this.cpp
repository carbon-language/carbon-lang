// RUN: %clang_cc1 -fsyntax-only -verify %s 
int x = this; // expected-error {{error: invalid use of 'this' outside of a nonstatic member function}}

void f() {
  int x = this; // expected-error {{error: invalid use of 'this' outside of a nonstatic member function}}
}
