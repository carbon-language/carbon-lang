// RUN: %clang_cc1 -fsyntax-only -Wunused-label -verify %s

void f() {
  a:
  goto a;
  b: // expected-warning{{unused}}
  c: __attribute__((unused));    // expected-warning {{unused label 'c'}}
  d: __attribute__((noreturn)); // expected-warning {{the only valid attribute for labels is 'unused'}}
  goto d;
  return;
}
