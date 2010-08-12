// RUN: %clang_cc1 -analyze -analyzer-store=region -analyzer-constraints=range -fblocks -verify -analyzer-opt-analyze-nested-blocks -analyzer-check-objc-mem -analyzer-check-idempotent-operations -verify %s

// Basic tests

extern void test(int i);
extern void test_f(float f);

void basic() {
  int x = 10, zero = 0, one = 1;

  // x op x
  x = x;        // expected-warning {{Assigned value is always the same as the existing value}}
  test(x - x);  // expected-warning {{Both operands to '-' always have the same value}}
  x -= x;       // expected-warning {{Both operands to '-=' always have the same value}}
  x = 10;       // no-warning
  test(x / x);  // expected-warning {{Both operands to '/' always have the same value}}
  x /= x;       // expected-warning {{Both operands to '/=' always have the same value}}
  x = 10;       // no-warning
  test(x & x);  // expected-warning {{Both operands to '&' always have the same value}}
  x &= x;       // expected-warning {{Both operands to '&=' always have the same value}}
  test(x | x);  // expected-warning {{Both operands to '|' always have the same value}}
  x |= x;       // expected-warning {{Both operands to '|=' always have the same value}}

  // x op 1
  test(x * one);  // expected-warning {{The right operand to '*' is always 1}}
  x *= one;       // expected-warning {{The right operand to '*=' is always 1}}
  test(x / one);  // expected-warning {{The right operand to '/' is always 1}}
  x /= one;       // expected-warning {{The right operand to '/=' is always 1}}

  // 1 op x
  test(one * x);   // expected-warning {{The left operand to '*' is always 1}}

  // x op 0
  test(x + zero);  // expected-warning {{The right operand to '+' is always 0}}
  test(x - zero);  // expected-warning {{The right operand to '-' is always 0}}
  test(x * zero);  // expected-warning {{The right operand to '*' is always 0}}
  test(x & zero);  // expected-warning {{The right operand to '&' is always 0}}
  test(x | zero);  // expected-warning {{The right operand to '|' is always 0}}
  test(x ^ zero);  // expected-warning {{The right operand to '^' is always 0}}
  test(x << zero); // expected-warning {{The right operand to '<<' is always 0}}
  test(x >> zero); // expected-warning {{The right operand to '>>' is always 0}}

  // 0 op x
  test(zero + x);  // expected-warning {{The left operand to '+' is always 0}}
  test(zero - x);  // expected-warning {{The left operand to '-' is always 0}}
  test(zero / x);  // expected-warning {{The left operand to '/' is always 0}}
  test(zero * x);  // expected-warning {{The left operand to '*' is always 0}}
  test(zero & x);  // expected-warning {{The left operand to '&' is always 0}}
  test(zero | x);  // expected-warning {{The left operand to '|' is always 0}}
  test(zero ^ x);  // expected-warning {{The left operand to '^' is always 0}}
  test(zero << x); // expected-warning {{The left operand to '<<' is always 0}}
  test(zero >> x); // expected-warning {{The left operand to '>>' is always 0}}
}

void floats(float x) {
  test_f(x * 1.0);  // no-warning
  test_f(x * 1.0F); // no-warning
}

// Ensure that we don't report false poitives in complex loops
void bailout() {
  int unused = 0, result = 4;
  result = result; // expected-warning {{Assigned value is always the same as the existing value}}

  for (unsigned bg = 0; bg < 1024; bg ++) {
    result = bg * result; // no-warning

    for (int i = 0; i < 256; i++) {
      unused *= i; // no-warning
    }
  }
}

// False positive tests

unsigned false1() {
  return (5 - 2 - 3); // no-warning
}

enum testenum { enum1 = 0, enum2 };
unsigned false2() {
  return enum1; // no-warning
}

extern unsigned foo();
