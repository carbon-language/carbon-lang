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
  int t13 = va_arg(ap, int);
}
