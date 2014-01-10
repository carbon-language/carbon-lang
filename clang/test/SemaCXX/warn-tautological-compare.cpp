// Force x86-64 because some of our heuristics are actually based
// on integer sizes.

// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -verify -std=c++11 %s

namespace RuntimeBehavior {
  // Avoid emitting tautological compare warnings when the code already has
  // compile time checks on variable sizes.

  const int kintmax = 2147483647;
  void test0(short x) {
    if (sizeof(x) < sizeof(int) || x < kintmax) {}

    if (x < kintmax) {}
    // expected-warning@-1{{comparison of constant 2147483647 with expression of type 'short' is always true}}
  }

  void test1(short x) {
    if (x < kintmax) {}
    // expected-warning@-1{{comparison of constant 2147483647 with expression of type 'short' is always true}}

    if (sizeof(x) < sizeof(int))
      return;

    if (x < kintmax) {}
  }
}
