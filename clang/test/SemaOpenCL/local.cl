// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

__kernel void foo(void) {
  __local int i;
  __local int j = 2; // expected-error {{'__local' variable cannot have an initializer}}
}
