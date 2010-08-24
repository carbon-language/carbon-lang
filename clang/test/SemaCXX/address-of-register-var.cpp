// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar // 8018252

void f0() {
  extern void f0_1(int*);
  register int x;
  f0_1(&x);
}

