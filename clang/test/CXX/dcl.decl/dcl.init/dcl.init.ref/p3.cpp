// RUN: clang-cc -fsyntax-only -verify %s
int& r1;	// expected-error{{declaration of reference variable 'r1' requires an initializer}}
extern int& r2;
