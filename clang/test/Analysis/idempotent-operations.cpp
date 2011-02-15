// RUN: %clang_cc1 -analyze -analyzer-store=region -analyzer-constraints=range -fblocks -analyzer-opt-analyze-nested-blocks -analyzer-check-objc-mem -analyzer-checker=core.experimental.IdempotentOps -verify %s

// C++ specific false positives

extern void test(int i);
extern void test_ref(int &i);

// Test references affecting pseudoconstants
void false1() {
  int a = 0;
  int five = 5;
  int &b = a;
   test(five * a); // expected-warning {{The right operand to '*' is always 0}}
   b = 4;
}
