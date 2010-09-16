// RUN: %clang_cc1 -fsyntax-only -verify %s

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

// Don't warn about an empty body if is expanded from a macro.
void f4(int i) {
  #define BODY ;
  if (i == i) // expected-warning{{self-comparison always evaluates to true}}
    BODY
  #undef BODY
}

