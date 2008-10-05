//-*- C -*-

/* This is a 
   multiline comment */

// Intended to exercise all syntactic parts of the C language.

int g0;
int g1, g2;

struct s0;

struct s0 {
  int x;
};

int g3 = 10;

__asm("");

typedef int td0;

td0 g4;

enum e0 {
  ec0
};

static void f0(int x) {
}

inline void f0_0(int x) {
  ;
}

extern void f0_1(int x) {
}

void f1(int, ...);

// Statements.
void f2() {
  for (;;) {
    break;
    continue;
  }

  while (0) {
  }

  do {
  } while (0);

  void *label = &&theif;
  goto *label;

  goto theif;
theif:
  if (0) {
    ;
  } else if (0) {
  } else {
  }

  switch(0) {
  case 0:
  case 1 ... 2:
    break;
  default:
    break;
  }

  asm ("nop");

  return;
}

// Expressions.

#include <stdarg.h>

typedef struct ipair {
  int first, second;
} ipair;

void f4(int a0, int a1, int a2, va_list ap) {
  int t0 = a0 ? a1 : a2;
  float t1 = (float) a0;
  ipair t2 = {1, 2};
  int t3 = sizeof(ipair);
  ipair t4;
  t4 = (ipair) {1, 2};
  extern int g(int);
  int t5 = g(a0);
  int t6 = t4.first;
  int t7[10];
  int t8 = t7[a0];
  t8++;
  const char *t9 = __FUNCTION__;
  char t10 = 'x';
  int t11 = __builtin_offsetof(ipair, first);
  int t12 = __builtin_types_compatible_p(ipair, int);  
  int t12_0 = __builtin_classify_type(t0);
  int t12_1 = __builtin_classify_type(t1);
  int t12_2 = __builtin_classify_type(t2);
  int t13 = va_arg(ap, int);
  va_list t13_0;
  va_copy(t13_0, ap);
  int t14 = __extension__(t13);
  int t15 = +t13;
  unsigned t16 = t14 ^ t15;
  int t17 = t14 % t15;
  int t17_0 = t16 % t16;
  float t18;
  int t19 = t18 ? 0 : 1;
  char *t20; ++t20; --t20;
  float t21; ++t21; --t21;
  double t22; ++t22; --t22;
  long double t23; ++t23; --t23;
  int t24 = !t19;
  int t25 = __real t24;
  int t26 = __imag t24;
  const char *t27 = t9;
  t27 += (unsigned char) 0xFF;
  t27 += (signed char) 0xFF;
  
  struct { char f0[10]; } *t28;
  int t29 = t28 - t28;
}

// Extended vectors

typedef __attribute__((ext_vector_type(4))) float float4;

void f5() {
  float4 t0 = (float4) { 0, 1, 2, 3 };
  float4 t1 = t0;
  t0.lo.e = t1.hi.x;
}
