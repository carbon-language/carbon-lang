// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-check-dead-stores -verify -Wno-unreachable-code -analyzer-opt-analyze-nested-blocks -analyzer-stats %s

int foo();

int test() { // expected-warning{{Total CFGBlocks}}
  int a = 1;
  a = 34 / 12;

  if (foo())
    return a;

  a /= 4;
  return a;
}
