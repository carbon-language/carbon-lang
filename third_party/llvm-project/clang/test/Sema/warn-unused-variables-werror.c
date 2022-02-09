// RUN: %clang_cc1 -fsyntax-only -Wunused-variable -Werror -verify %s

void f() {
  int i;  // expected-error{{unused}}
  int j;  // expected-error{{unused}}
}
