// RUN: %clang_cc1 -fsyntax-only -verify -Wtautological-compare %s

#define mydefine 2

void f(int x) {
  if ((8 & x) == 3) {}  // expected-warning {{bitwise comparison always evaluates to false}}
  if ((x & 8) == 4) {}  // expected-warning {{bitwise comparison always evaluates to false}}
  if ((x & 8) != 4) {}  // expected-warning {{bitwise comparison always evaluates to true}}
  if ((2 & x) != 4) {}  // expected-warning {{bitwise comparison always evaluates to true}}
  if ((x | 4) == 3) {}  // expected-warning {{bitwise comparison always evaluates to false}}
  if ((x | 3) != 4) {}  // expected-warning {{bitwise comparison always evaluates to true}}
  if ((5 | x) != 3) {}  // expected-warning {{bitwise comparison always evaluates to true}}
  if ((x & 0x15) == 0x13) {} // expected-warning {{bitwise comparison always evaluates to false}}
  if ((0x23 | x) == 0x155){} // expected-warning {{bitwise comparison always evaluates to false}}

  if ((x & 8) == 8) {}
  if ((x & 8) != 8) {}
  if ((x | 4) == 4) {}
  if ((x | 4) != 4) {}

  if ((x & 9) == 8) {}
  if ((x & 9) != 8) {}
  if ((x | 4) == 5) {}
  if ((x | 4) != 5) {}

  if ((x & mydefine) == 8) {}
  if ((x | mydefine) == 4) {}
}
