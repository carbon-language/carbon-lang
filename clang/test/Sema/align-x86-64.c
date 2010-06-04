// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s

// PR5599

void frob(void *);

void foo(void) {
  float x[4];
  char y[__alignof__(x) == 16 ? 1 : -1];
  frob(y);
}
