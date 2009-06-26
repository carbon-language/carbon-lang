// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x
void f() {
  auto int a; // expected-error{{cannot combine with previous 'auto' declaration specifier}}
  int auto b; // expected-error{{cannot combine with previous 'int' declaration specifier}}
}
