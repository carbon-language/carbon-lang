// RUN: %clang_cc1 -fsyntax-only -verify %s

void cat0(int a[static 0]) {} // expected-warning {{'static' has no effect on zero-length arrays}}

void cat(int a[static 3]) {} // expected-note 2 {{callee declares array parameter as static here}}

typedef int i3[static 3];
void tcat(i3 a) {}

void vat(int i, int a[static i]) {} // expected-note {{callee declares array parameter as static here}}

void f(int *p) {
  int a[2], b[3], c[4];

  cat0(0);

  cat(0); // expected-warning {{null passed to a callee which requires a non-null argument}}
  cat(a); // expected-warning {{array argument is too small; contains 2 elements, callee requires at least 3}}
  cat(b);
  cat(c);
  cat(p);

  tcat(0); // expected-warning {{null passed to a callee which requires a non-null argument}}
  tcat(a); // expected-warning {{array argument is too small; contains 2 elements, callee requires at least 3}}
  tcat(b);
  tcat(c);
  tcat(p);

  vat(1, 0); // expected-warning {{null passed to a callee which requires a non-null argument}}
  vat(3, b);
}
