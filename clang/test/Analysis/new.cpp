// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem -analyzer-store region -verify %s

void f1() {
  int *n1 = new int;
  if (*n1) { // expected-warning {{Branch condition evaluates to a garbage value}}
  }

  int *n2 = new int(3);
  if (*n2) { // no-warning
  }
}

