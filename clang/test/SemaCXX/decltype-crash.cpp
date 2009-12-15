// RUN: %clang_cc1 -fsyntax-only -verify %s

int& a();

void f() {
  decltype(a()) c; // expected-error {{no matching function for call to 'decltype'}}
}
