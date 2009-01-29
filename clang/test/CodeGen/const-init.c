// RUN: clang -verify -emit-llvm -o %t %s &&

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

// RUN: grep '@g1x = global { double, double } { double 1.000000e+00, double 0.000000e+00 }' %t &&
_Complex double g1x = 1.0f;
// RUN: grep '@g1y = global { double, double } { double 0.000000e+00, double 1.000000e+00 }' %t &&
_Complex double g1y = 1.0fi;
// RUN: grep '@g1 = global { i8, i8 } { i8 1, i8 10 }' %t &&
_Complex char g1 = (char) 1 + (char) 10 * 1i;
// RUN: grep '@g2 = global { i32, i32 } { i32 1, i32 10 }' %t &&
_Complex int g2 = 1 + 10i;
// RUN: grep '@g3 = global { float, float } { float 1.000000e+00, float 1.000000e+01 }' %t &&
_Complex float g3 = 1.0 + 10.0i;
// RUN: grep '@g4 = global { double, double } { double 1.000000e+00, double 1.000000e+01 }' %t &&
_Complex double g4 = 1.0 + 10.0i;
// RUN: grep '@g5 = global { i32, i32 } zeroinitializer' %t &&
_Complex int g5 = (2 + 3i) == (5 + 7i);
// RUN: grep '@g6 = global { double, double } { double -1.100000e+01, double 2.900000e+01 }' %t &&
_Complex double g6 = (2.0 + 3.0i) * (5.0 + 7.0i);
// RUN: grep '@g7 = global i32 1' %t &&
int g7 = (2 + 3i) * (5 + 7i) == (-11 + 29i);
// RUN: grep '@g8 = global i32 1' %t &&
int g8 = (2.0 + 3.0i) * (5.0 + 7.0i) == (-11.0 + 29.0i);
// RUN: grep '@g9 = global i32 0' %t &&
int g9 = (2 + 3i) * (5 + 7i) != (-11 + 29i);
// RUN: grep '@g10 = global i32 0' %t &&
int g10 = (2.0 + 3.0i) * (5.0 + 7.0i) != (-11.0 + 29.0i);


// RUN: true
