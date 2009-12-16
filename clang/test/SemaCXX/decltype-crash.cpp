// RUN: %clang_cc1 -fsyntax-only -verify %s

int& a();

void f() {
  decltype(a()) c; // expected-error {{use of undeclared identifier 'decltype'}}
}
