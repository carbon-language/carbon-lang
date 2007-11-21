// RUN: clang -fsyntax-only -verify %s

void f1() {
  asm ("ret" : : :); // expected-error {{expected string literal}}
}
