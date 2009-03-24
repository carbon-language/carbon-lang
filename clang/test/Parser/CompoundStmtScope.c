// RUN: clang-cc -fsyntax-only -verify %s

int foo() {
  {
    typedef float X;
  }
  X Y;  // expected-error {{use of undeclared identifier}}
}
