// RUN: %clang_cc1 -fsyntax-only -verify %s 

void choice(int);
int choice(bool);

void test() {
  // Result of ! must be type bool.
  int i = choice(!1);
}

// rdar://8018252
void f0() {
  extern void f0_1(int*);
  register int x;
  f0_1(&x);
}
