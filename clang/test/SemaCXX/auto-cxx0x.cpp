// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x
void f() {
  auto int a; // expected-warning {{'auto' storage class specifier is not permitted in C++0x, and will not be supported in future releases}}
  int auto b; // expected-error{{cannot combine with previous 'int' declaration specifier}}
}
