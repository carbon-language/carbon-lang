// RUN: %clang_cc1 -fsyntax-only -verify %s

void f() {
  void *p = malloc(sizeof(int) * 10); // expected-error{{no matching function for call to 'malloc'}}
}

int malloc(double);
