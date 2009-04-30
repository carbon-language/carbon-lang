// RUN: clang-cc %s -fsyntax-only -verify

// clang emits the following warning by default.
// With GCC, -pedantic, -Wreturn-type or -Wall are required to produce the 
// following warning.
int t14() {
  return; // expected-warning {{non-void function 't14' should return a value}}
}

