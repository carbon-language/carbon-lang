// RUN: %clang_cc1 %s -verify -fsyntax-only

void f(_Atomic(int) a, _Atomic(int) b) {
  if (a > b)      {} // no warning
  if (a < b)      {} // no warning
  if (a >= b)     {} // no warning
  if (a <= b)     {} // no warning
  if (a == b)     {} // no warning
  if (a != b)     {} // no warning

  if (a == 0) {} // no warning
  if (a > 0) {} // no warning
  if (a > 1) {} // no warning
  if (a > 2) {} // no warning

  if (!a > 0) {}  // no warning
  if (!a > 1)     {} // expected-warning {{comparison of constant 1 with boolean expression is always false}}
  if (!a > 2)     {} // expected-warning {{comparison of constant 2 with boolean expression is always false}}
  if (!a > b)     {} // no warning
  if (!a > -1)    {} // expected-warning {{comparison of constant -1 with boolean expression is always true}}
}
