// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -analyzer-config alpha.clone.CloneChecker:MinimumCloneComplexity=10 -verify %s

// expected-no-diagnostics


int foo1(int a, int b) {
  if (a > b)
    return a;
  return b;
}

// Different types, so not a clone
int foo2(long a, long b) {
  if (a > b)
    return a;
  return b;
}
