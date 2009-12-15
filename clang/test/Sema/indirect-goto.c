// RUN: %clang_cc1 -fsyntax-only -verify %s

struct c {int x;};
int a(struct c x, long long y) {
  goto *x; // expected-error{{incompatible type}}
  goto *y; // expected-warning{{incompatible integer to pointer conversion}}
}

