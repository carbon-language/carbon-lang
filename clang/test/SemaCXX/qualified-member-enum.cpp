// RUN: %clang_cc1 -fsyntax-only -verify %s 

// Check that this doesn't crash.
struct A {
  enum {LABEL};
};
int f() {
  return A().A::LABEL;
}

