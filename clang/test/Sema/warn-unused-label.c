// RUN: %clang_cc1 -fsyntax-only -Wunused-label -verify %s

void f() {
  a:
  goto a;
  b: // expected-warning{{unused}}
  return;
}
