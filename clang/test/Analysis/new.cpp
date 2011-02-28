// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-store region -verify %s

void f1() {
  int *n = new int;
  if (*n) { // expected-warning {{Branch condition evaluates to a garbage value}}
  }
}

void f2() {
  int *n = new int(3);
  if (*n) { // no-warning
  }
}

