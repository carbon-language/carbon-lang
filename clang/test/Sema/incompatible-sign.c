// RUN: %clang_cc1 %s -verify -fsyntax-only

int a(int* x);
int b(unsigned* y) { return a(y); } // expected-warning {{passing 'unsigned int *' to parameter of type 'int *' converts between pointers to integer types with different sign}}

