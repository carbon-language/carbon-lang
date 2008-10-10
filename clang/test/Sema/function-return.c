// RUN: clang %s -fsyntax-only -verify -pedantic
// PR2790

void f1() {
  return 0; // expected-warning {{void function 'f1' should not return a value}}
}

int f2() {
  return; // expected-warning {{non-void function 'f2' should return a value}}
}
