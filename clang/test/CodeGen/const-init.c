// RUN: clang -verify -emit-llvm -o %t %s

#include <stdint.h>

// Brace-enclosed string array initializers
char a[] = { "asdf" };

// Double-implicit-conversions of array/functions (not legal C, but
// clang accepts it for gcc compat).
intptr_t b = a; // expected-warning {{incompatible pointer to integer conversion}}
int c();
void *d = c;
intptr_t e = c; // expected-warning {{incompatible pointer to integer conversion}}

int f, *g = __extension__ &f, *h = (1 != 1) ? &f : &f;

union s2 {
  struct {
    struct { } *f0;
  } f0;
};

int g0 = (int)(&(((union s2 *) 0)->f0.f0) - 0);

_Complex int g1 = 1 + 10i;
_Complex double g2 = 1.0 + 10.0i;
