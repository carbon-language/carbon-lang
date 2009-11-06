// RUN: clang-cc -fsyntax-only -pedantic -verify -Wsign-compare %s

int test0(long a, unsigned long b) {
  enum Enum {B};
  return (a == B) +        // expected-warning {{comparison of integers of different signs}}
         ((int)a == B) +   // expected-warning {{comparison of integers of different signs}}
         ((short)a == B) + // expected-warning {{comparison of integers of different signs}}
         (a == (unsigned int) B) +  // expected-warning {{comparison of integers of different signs}}
         (a == (unsigned short) B); // expected-warning {{comparison of integers of different signs}}         

  // Should be able to prove all of these are non-negative.
  return (b == (long) B) +
         (b == (int) B) +
         (b == (short) B);
}
