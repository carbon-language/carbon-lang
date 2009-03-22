// RUN: clang %s -verify -fsyntax-only

int a(int* x);
int b(unsigned* y) { return a(y); } // expected-warning {{pointer types point to integer types with different sign}}

