// RUN: %clang_cc1 -fsyntax-only -verify -Wswitch-enum -Wno-switch-enum-redundant-default %s

int test1() {
  enum { A, B } a;
  switch (a) { //expected-warning{{enumeration value 'B' not handled in switch}}
  case A: return 1;
  default: return 2;
  }
}

int test2() {
  enum { A, B } a;
  switch (a) {
  case A: return 1;
  case B: return 2;
  default: return 3;
  }
}
