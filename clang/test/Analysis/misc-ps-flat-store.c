// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-store=flat -verify %s

void f1() {
  int x;
  int *p;
  x = 1;
  p = 0;
  if (x != 1)
    *p = 1; // no-warning
}
