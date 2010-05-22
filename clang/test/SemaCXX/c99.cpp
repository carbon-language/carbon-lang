// RUN: %clang_cc1 -fsyntax-only -verify %s
void f1(int i[static 5]) { // expected-error{{C99}}
}
