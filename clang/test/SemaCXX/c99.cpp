// RUN: %clang_cc1 -fsyntax-only -verify %s

void f0(int i) {
  char array[i]; // expected-error{{variable length arrays}}
}

void f1(int i[static 5]) { // expected-error{{C99}}
}
