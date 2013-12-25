// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -x c %s

// Test that GNU C extension __builtin_types_compatible_p() is not available in C++ mode.

int f() {
  return __builtin_types_compatible_p(int, const int); // expected-error{{}}
}
