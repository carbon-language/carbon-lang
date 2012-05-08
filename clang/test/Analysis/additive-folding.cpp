// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.deadcode.UnreachableCode,unix.Malloc -verify -analyzer-constraints=basic %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.deadcode.UnreachableCode,unix.Malloc -verify -analyzer-constraints=range %s

// These are used to trigger warnings.
typedef typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
#define NULL ((void*)0)
#define UINT_MAX (~0U)
#define INT_MAX (UINT_MAX & (UINT_MAX >> 1))
#define INT_MIN (-INT_MAX - 1)

//---------------
//  Plus/minus
//---------------

void separateExpressions (int a) {
  int b = a + 1;
  --b;

  void *buf = malloc(1);
  if (a != 0 && b == 0)
    return; // expected-warning{{never executed}}
  free(buf);
}

void oneLongExpression (int a) {
  // Expression canonicalization should still allow this to work, even though
  // the first term is on the left.
  int b = 15 + a + 15 - 10 - 20;

  void *buf = malloc(1);
  if (a != 0 && b == 0)
    return; // expected-warning{{never executed}}
  free(buf);
}

void mixedTypes (int a) {
  void *buf = malloc(1);

  // Different additive types should not cause crashes when constant-folding.
  // This is part of PR7406.
  int b = a + 1LL;
  if (a != 0 && (b-1) == 0) // not crash
    return; // expected-warning{{never executed}}

  int c = a + 1U;
  if (a != 0 && (c-1) == 0) // not crash
    return; // expected-warning{{never executed}}

  free(buf);
}

//---------------
//  Comparisons
//---------------

// Equality and inequality only
void eq_ne (unsigned a) {
  void *b = NULL;
  if (a == UINT_MAX)
    b = malloc(1);
  if (a+1 != 0)
    return; // no-warning
  if (a-1 != UINT_MAX-1)
    return; // no-warning
  free(b);
}

void ne_eq (unsigned a) {
  void *b = NULL;
  if (a != UINT_MAX)
    b = malloc(1);
  if (a+1 == 0)
    return; // no-warning
  if (a-1 == UINT_MAX-1)
    return; // no-warning
  free(b);
}

// Mixed typed inequalities (part of PR7406)
// These should not crash.
void mixed_eq_ne (int a) {
  void *b = NULL;
  if (a == 1)
    b = malloc(1);
  if (a+1U != 2)
    return; // no-warning
  if (a-1U != 0)
    return; // expected-warning{{never executed}}
  free(b);
}

void mixed_ne_eq (int a) {
  void *b = NULL;
  if (a != 1)
    b = malloc(1);
  if (a+1U == 2)
    return; // no-warning
  if (a-1U == 0)
    return; // expected-warning{{never executed}}
  free(b);
}


// Simple order comparisons with no adjustment
void baselineGT (unsigned a) {
  void *b = NULL;
  if (a > 0)
    b = malloc(1);
  if (a == 0)
    return; // no-warning
  free(b);
}

void baselineGE (unsigned a) {
  void *b = NULL;
  if (a >= UINT_MAX)
    b = malloc(1);
  if (a == UINT_MAX)
    free(b);
  return; // no-warning
}

void baselineLT (unsigned a) {
  void *b = NULL;
  if (a < UINT_MAX)
    b = malloc(1);
  if (a == UINT_MAX)
    return; // no-warning
  free(b);
}

void baselineLE (unsigned a) {
  void *b = NULL;
  if (a <= 0)
    b = malloc(1);
  if (a == 0)
    free(b);
  return; // no-warning
}


// Adjustment gives each of these an extra solution!
void adjustedGT (unsigned a) {
  void *b = NULL;
  if (a-1 > UINT_MAX-1)
    b = malloc(1);
  return; // expected-warning{{leak}}
}

void adjustedGE (unsigned a) {
  void *b = NULL;
  if (a-1 >= UINT_MAX-1)
    b = malloc(1);
  if (a == UINT_MAX)
    free(b);
  return; // expected-warning{{leak}}
}

void adjustedLT (unsigned a) {
  void *b = NULL;
  if (a+1 < 1)
    b = malloc(1);
  return; // expected-warning{{leak}}
}

void adjustedLE (unsigned a) {
  void *b = NULL;
  if (a+1 <= 1)
    b = malloc(1);
  if (a == 0)
    free(b);
  return; // expected-warning{{leak}}
}


// Tautologies
void tautologyGT (unsigned a) {
  void *b = malloc(1);
  if (a > UINT_MAX)
    return; // no-warning
  free(b);
}

void tautologyGE (unsigned a) {
  void *b = malloc(1);
  if (a >= 0) // expected-warning{{always true}}
    free(b);
  return; // no-warning
}

void tautologyLT (unsigned a) {
  void *b = malloc(1);
  if (a < 0) // expected-warning{{always false}}
    return; // expected-warning{{never executed}}
  free(b);
}

void tautologyLE (unsigned a) {
  void *b = malloc(1);
  if (a <= UINT_MAX)
    free(b);
  return; // no-warning
}


// Tautologies from outside the range of the symbol
void tautologyOutsideGT(unsigned char a) {
  void *b = malloc(1);
  if (a > 0x100)
    return; // expected-warning{{never executed}}
  if (a > -1)
    free(b);
  return; // no-warning
}

void tautologyOutsideGE(unsigned char a) {
  void *b = malloc(1);
  if (a >= 0x100)
    return; // expected-warning{{never executed}}
  if (a >= -1)
    free(b);
  return; // no-warning
}

void tautologyOutsideLT(unsigned char a) {
  void *b = malloc(1);
  if (a < -1)
    return; // expected-warning{{never executed}}
  if (a < 0x100)
    free(b);
  return; // no-warning
}

void tautologyOutsideLE (unsigned char a) {
  void *b = malloc(1);
  if (a <= -1)
    return; // expected-warning{{never executed}}
  if (a <= 0x100)
    free(b);
  return; // no-warning
}

void tautologyOutsideEQ(unsigned char a) {
  if (a == 0x100)
    malloc(1); // expected-warning{{never executed}}
  if (a == -1)
    malloc(1); // expected-warning{{never executed}}
}

void tautologyOutsideNE(unsigned char a) {
  void *sentinel = malloc(1);
  if (a != 0x100)
    free(sentinel);

  sentinel = malloc(1);
  if (a != -1)
    free(sentinel);
}


// Wraparound with mixed types. Note that the analyzer assumes
// -fwrapv semantics.
void mixedWraparoundSanityCheck(int a) {
  int max = INT_MAX;
  int min = INT_MIN;

  int b = a + 1;
  if (a == max && b != min)
    return; // expected-warning{{never executed}}
}

void mixedWraparoundGT(int a) {
  int max = INT_MAX;

  if ((a + 2) > (max + 1LL))
    return; // expected-warning{{never executed}}
}

void mixedWraparoundGE(int a) {
  int max = INT_MAX;
  int min = INT_MIN;

  if ((a + 2) >= (max + 1LL))
    return; // expected-warning{{never executed}}

  void *sentinel = malloc(1);
  if ((a - 2LL) >= min)
    free(sentinel);
  return; // expected-warning{{leak}}
}

void mixedWraparoundLT(int a) {
  int min = INT_MIN;

  if ((a - 2) < (min - 1LL))
    return; // expected-warning{{never executed}}
}

void mixedWraparoundLE(int a) {
  int max = INT_MAX;
  int min = INT_MIN;

  if ((a - 2) <= (min - 1LL))
    return; // expected-warning{{never executed}}

  void *sentinel = malloc(1);
  if ((a + 2LL) <= max)
    free(sentinel);
  return; // expected-warning{{leak}}
}

void mixedWraparoundEQ(int a) {
  int max = INT_MAX;

  if ((a + 2) == (max + 1LL))
    return; // expected-warning{{never executed}}
}

void mixedWraparoundNE(int a) {
  int max = INT_MAX;

  void *sentinel = malloc(1);
  if ((a + 2) != (max + 1LL))
    free(sentinel);
  return; // no-warning
}


// Mixed-signedness comparisons.
void mixedSignedness(int a, unsigned b) {
  int sMin = INT_MIN;
  unsigned uMin = INT_MIN;
  if (a == sMin && a != uMin)
    return; // expected-warning{{never executed}}
  if (b == uMin && b != sMin)
    return; // expected-warning{{never executed}}
}


// PR12206/12510 - When SimpleSValBuilder figures out that a symbol is fully
// constrained, it should cast the value to the result type in a binary
// operation...unless the binary operation is a comparison, in which case the
// two arguments should be the same type, but won't match the result type.
//
// This is easier to trigger in C++ mode, where the comparison result type is
// 'bool' and is thus differently sized from int on pretty much every system.
//
// This is not directly related to additive folding, but we use SValBuilder's
// additive folding to tickle the bug. ExprEngine will simplify fully-constrained
// symbols, so SValBuilder will only see them if they are (a) part of an evaluated
// SymExpr (e.g. with additive folding) or (b) generated by a checker (e.g.
// unix.cstring's strlen() modelling).
void PR12206(int x) {
  // Build a SymIntExpr, dependent on x.
  int local = x - 1;

  // Constrain the value of x.
  int value = 1 + (1 << (8 * sizeof(1 == 1))); // not representable by bool
  if (x != value) return;

  // Constant-folding will turn (local+1) back into the symbol for x.
  // The point of this dance is to make SValBuilder be responsible for
  // turning the symbol into a ConcreteInt, rather than ExprEngine.

  // Test relational operators.
  if ((local + 1) < 2)
    malloc(1); // expected-warning{{never executed}}
  if (2 > (local + 1))
    malloc(1); // expected-warning{{never executed}}

  // Test equality operators.
  if ((local + 1) == 1) 
    malloc(1); // expected-warning{{never executed}}
  if (1 == (local + 1))
    malloc(1); // expected-warning{{never executed}}
}

void PR12206_truncation(signed char x) {
  // Build a SymIntExpr, dependent on x.
  signed char local = x - 1;

  // Constrain the value of x.
  if (x != 1) return;

  // Constant-folding will turn (local+1) back into the symbol for x.
  // The point of this dance is to make SValBuilder be responsible for
  // turning the symbol into a ConcreteInt, rather than ExprEngine.

  // Construct a value that cannot be represented by 'char',
  // but that has the same lower bits as x.
  signed int value = 1 + (1 << 8);

  // Test relational operators.
  if ((local + 1) >= value)
    malloc(1); // expected-warning{{never executed}}
  if (value <= (local + 1))
    malloc(1); // expected-warning{{never executed}}

  // Test equality operators.
  if ((local + 1) == value) 
    malloc(1); // expected-warning{{never executed}}
  if (value == (local + 1))
    malloc(1); // expected-warning{{never executed}}
}

void multiplicativeSanityTest(int x) {
  // At one point we were ignoring the *4 completely -- the constraint manager
  // would see x < 8 and then declare the next part unreachable.
  if (x*4 < 8)
    return;
  if (x == 3)
    malloc(1);
  return; // expected-warning{{leak}}
}
