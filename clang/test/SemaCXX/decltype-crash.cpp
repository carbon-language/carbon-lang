// RUN: clang-cc -fsyntax-only -verify %s

int& a();

void f() {
  decltype(a()) c; // expected-error {{no matching function for call to 'decltype'}}
}
