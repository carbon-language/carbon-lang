// RUN: %clang_cc1 -triple i386-mingw32 -fms-extensions -fsyntax-only -verify %s

int foo(int *a, int i) {
  __assume(i != 4);
  __assume(++i > 2); //expected-warning {{the argument to __assume has side effects that will be discarded}}

  int test = sizeof(struct{char qq[(__assume(i != 5), 7)];});

  return a[i];
}

