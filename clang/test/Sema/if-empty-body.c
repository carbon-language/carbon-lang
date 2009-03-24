// RUN: clang-cc -fsyntax-only -verify %s

void f1(int a) {
    if (a); // expected-warning {{if statement has empty body}}
}

void f2(int a) {
    if (a) {}
}

void f3() {
  if (1)
    xx;      // expected-error {{use of undeclared identifier}}
  return;    // no empty body warning.
}

