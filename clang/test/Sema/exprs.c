// RUN: clang %s -verify -pedantic -fsyntax-only

// PR1966
_Complex double test1() {
  return __extension__ 1.0if;
}

_Complex double test2() {
  return 1.0if;    // expected-warning {{imaginary constants are an extension}}
}

// rdar://6097308
void test3() {
  int x;
  (__extension__ x) = 10;
}

