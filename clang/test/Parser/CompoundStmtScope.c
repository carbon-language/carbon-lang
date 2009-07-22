// RUN: clang-cc -fsyntax-only -verify %s

void foo() {
  {
    typedef float X;
  }
  X Y;  // expected-error {{use of undeclared identifier}}
}
