// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98
void f() {
  auto int a; // expected-warning {{'auto' storage class specifier is redundant and incompatible with C++0x}}
  int auto b; // expected-warning {{'auto' storage class specifier is redundant and incompatible with C++0x}}
  auto c; // expected-warning {{C++0x extension}} expected-error {{requires an initializer}}
  static auto d = 0; // expected-warning {{C++0x extension}}
  auto static e = 0; // expected-warning {{C++0x extension}}
}
