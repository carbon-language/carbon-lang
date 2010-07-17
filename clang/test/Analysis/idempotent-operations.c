// RUN: %clang_cc1 -analyze -analyzer-store=region -analyzer-constraints=range -fblocks -verify -analyzer-opt-analyze-nested-blocks -analyzer-check-objc-mem -verify %s

// Basic tests

extern void test(int i);
extern void test_f(float f);

void basic() {
  int x = 10, zero = 0, one = 1;

  // x op x
  x = x;        // expected-warning {{idempotent operation; both operands are always equal in value}}
  test(x - x);  // expected-warning {{idempotent operation; both operands are always equal in value}}
  x -= x;       // expected-warning {{idempotent operation; both operands are always equal in value}}
  x = 10;       // no-warning
  test(x / x);  // expected-warning {{idempotent operation; both operands are always equal in value}}
  x /= x;       // expected-warning {{idempotent operation; both operands are always equal in value}}
  x = 10;       // no-warning
  test(x & x);  // expected-warning {{idempotent operation; both operands are always equal in value}}
  x &= x;       // expected-warning {{idempotent operation; both operands are always equal in value}}
  test(x | x);  // expected-warning {{idempotent operation; both operands are always equal in value}}
  x |= x;       // expected-warning {{idempotent operation; both operands are always equal in value}}

  // x op 1
  test(x * one);  // expected-warning {{idempotent operation; the right operand is always 1}}
  x *= one;       // expected-warning {{idempotent operation; the right operand is always 1}}
  test(x / one);  // expected-warning {{idempotent operation; the right operand is always 1}}
  x /= one;       // expected-warning {{idempotent operation; the right operand is always 1}}

  // 1 op x
  test(one * x);   // expected-warning {{idempotent operation; the left operand is always 1}}

  // x op 0
  test(x + zero);  // expected-warning {{idempotent operation; the right operand is always 0}}
  test(x - zero);  // expected-warning {{idempotent operation; the right operand is always 0}}
  test(x * zero);  // expected-warning {{idempotent operation; the right operand is always 0}}
  test(x & zero);  // expected-warning {{idempotent operation; the right operand is always 0}}
  test(x | zero);  // expected-warning {{idempotent operation; the right operand is always 0}}
  test(x ^ zero);  // expected-warning {{idempotent operation; the right operand is always 0}}
  test(x << zero); // expected-warning {{idempotent operation; the right operand is always 0}}
  test(x >> zero); // expected-warning {{idempotent operation; the right operand is always 0}}

  // 0 op x
  test(zero + x);  // expected-warning {{idempotent operation; the left operand is always 0}}
  test(zero - x);  // expected-warning {{idempotent operation; the left operand is always 0}}
  test(zero / x);  // expected-warning {{idempotent operation; the left operand is always 0}}
  test(zero * x);  // expected-warning {{idempotent operation; the left operand is always 0}}
  test(zero & x);  // expected-warning {{idempotent operation; the left operand is always 0}}
  test(zero | x);  // expected-warning {{idempotent operation; the left operand is always 0}}
  test(zero ^ x);  // expected-warning {{idempotent operation; the left operand is always 0}}
  test(zero << x); // expected-warning {{idempotent operation; the left operand is always 0}}
  test(zero >> x); // expected-warning {{idempotent operation; the left operand is always 0}}
}

void floats(float x) {
  test_f(x * 1.0); // no-warning
  test_f(x * 1.0F); // no-warning
}
