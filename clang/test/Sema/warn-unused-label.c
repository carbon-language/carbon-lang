// RUN: %clang_cc1 -fsyntax-only -Wunused-label -verify %s

void f() {
  a:
  goto a;
  b: // expected-warning{{unused}}
  c: __attribute__((unused));
  d: __attribute__((noreturn)); // expected-warning {{The only valid attribute for labels is 'unused'}}
  goto d;
  return;
}
