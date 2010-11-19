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
  #define BODY(x)
  if (i == i) // expected-warning{{self-comparison always evaluates to true}}
    BODY(0);
  #undef BODY
}

template <typename T>
void tf() {
  #define BODY(x)
  if (0)
    BODY(0);
  #undef BODY
}

void f5() {
    tf<int>();
}
