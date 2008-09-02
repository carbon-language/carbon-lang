// RUN: clang -fsyntax-only -verify -pedantic %s

// The preprocessor shouldn't warn about extensions within macro bodies that
// aren't expanded.
#define __block __attribute__((__blocks__(byref)))

// This warning is entirely valid.
__block int x; // expected-warning{{extension used}}

void whatever() {}
