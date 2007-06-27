// RUN: clang -parse-ast-check %s

int foo() {
  {
    typedef float X;
  }
  X Y;  // expected-error {{use of undeclared identifier}}
}
