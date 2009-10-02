// RUN: clang-cc -triple i386-pc-linux-gnu -verify -emit-llvm -o %t %s | FileCheck %s --input-file=%t &&

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

// RUN: grep '@g1x = global %. { double 1.000000e+00, double 0.000000e+00 }' %t &&
_Complex double g1x = 1.0f;
// RUN: grep '@g1y = global %. { double 0.000000e+00, double 1.000000e+00 }' %t &&
_Complex double g1y = 1.0fi;
// RUN: grep '@g1 = global %. { i8 1, i8 10 }' %t &&
_Complex char g1 = (char) 1 + (char) 10 * 1i;
// RUN: grep '@g2 = global %2 { i32 1, i32 10 }' %t &&
_Complex int g2 = 1 + 10i;
// RUN: grep '@g3 = global %. { float 1.000000e+00, float 1.000000e+01 }' %t &&
_Complex float g3 = 1.0 + 10.0i;
// RUN: grep '@g4 = global %. { double 1.000000e+00, double 1.000000e+01 }' %t &&
_Complex double g4 = 1.0 + 10.0i;
// RUN: grep '@g5 = global %2 zeroinitializer' %t &&
_Complex int g5 = (2 + 3i) == (5 + 7i);
// RUN: grep '@g6 = global %. { double -1.100000e+01, double 2.900000e+01 }' %t &&
_Complex double g6 = (2.0 + 3.0i) * (5.0 + 7.0i);
// RUN: grep '@g7 = global i32 1' %t &&
int g7 = (2 + 3i) * (5 + 7i) == (-11 + 29i);
// RUN: grep '@g8 = global i32 1' %t &&
int g8 = (2.0 + 3.0i) * (5.0 + 7.0i) == (-11.0 + 29.0i);
// RUN: grep '@g9 = global i32 0' %t &&
int g9 = (2 + 3i) * (5 + 7i) != (-11 + 29i);
// RUN: grep '@g10 = global i32 0' %t &&
int g10 = (2.0 + 3.0i) * (5.0 + 7.0i) != (-11.0 + 29.0i);

// PR5108
// CHECK: @ss = global %4 <{ i32 0, i8 7 }>, align 1
struct s {
  unsigned long a;
  unsigned long b:3;
} __attribute__((__packed__)) ss  = { .a = 0x0, .b = 7,  };

// Global references
// RUN: grep '@g11.l0 = internal global i32 ptrtoint (i32 ()\* @g11 to i32)' %t &&
long g11() { 
  static long l0 = (long) g11;
  return l0; 
}

// RUN: grep '@g12 = global i32 ptrtoint (i8\* @g12_tmp to i32)' %t &&
static char g12_tmp;
long g12 = (long) &g12_tmp;

// RUN: grep '@g13 = global \[1 x %.truct.g13_s0\] \[%.truct.g13_s0 { i32 ptrtoint (i8\* @g12_tmp to i32) }\]' %t &&
struct g13_s0 {
   long a;
};
struct g13_s0 g13[] = {
   { (long) &g12_tmp }
};

// RUN: grep '@g14 = global i8\* inttoptr (i64 100 to i8\*)' %t &&
void *g14 = (void*) 100;

// RUN: grep '@g15 = global i32 -1' %t &&
int g15 = (int) (char) ((void*) 0 + 255);

// RUN: grep '@g16 = global i64 4294967295' %t &&
long long g16 = (long long) ((void*) 0xFFFFFFFF);

// RUN: grep '@g17 = global i32\* @g15' %t &&
int *g17 = (int *) ((long) &g15);

// RUN: grep '@g18.p = internal global \[1 x i32\*\] \[i32\* @g19\]' %t &&
void g18(void) {
  extern int g19;
  static int *p[] = { &g19 };
}

// RUN: grep '@g20.l0 = internal global %.truct.g20_s1 { %.truct.g20_s0\* null, %.truct.g20_s0\*\* getelementptr inbounds (%.truct.g20_s1\* @g20.l0, i32 0, i32 0) }' %t &&

struct g20_s0;
struct g20_s1 {
  struct g20_s0 *f0, **f1;
};
void *g20(void) {
  static struct g20_s1 l0 = { ((void*) 0), &l0.f0 };
  return l0.f1;
}

// PR4108
struct g21 {int g21;};
const struct g21 g21 = (struct g21){1};

// RUN: true

