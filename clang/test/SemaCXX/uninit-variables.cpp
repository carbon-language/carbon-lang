// RUN: %clang_cc1 -fsyntax-only -Wuninitialized-experimental -fsyntax-only %s -verify

int test1_aux(int &x);
int test1() {
  int x;
  test1_aux(x);
  return x; // no-warning
}

int test2_aux() {
  int x;
  int &y = x;
  return x; // no-warning
}

