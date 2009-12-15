// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x
void f() {
  auto int a; // expected-error{{cannot combine with previous 'auto' declaration specifier}} // expected-error{{declaration of variable 'a' with type 'auto' requires an initializer}}
  int auto b; // expected-error{{cannot combine with previous 'int' declaration specifier}}
}
