// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x
void f() {
  auto int a; // expected-warning {{'auto' storage class specifier is redundant and will be removed in future releases}}
  int auto b; // expected-error{{cannot combine with previous 'int' declaration specifier}}
}
