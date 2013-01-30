// RUN: %clang_cc1 -verify %s

kernel void no_ptrptr(global int **i) { } // expected-error{{kernel argument cannot be declared as a pointer to a pointer}}

kernel int bar()  { // expected-error {{kernel must have void return type}}
  return 6;
}
