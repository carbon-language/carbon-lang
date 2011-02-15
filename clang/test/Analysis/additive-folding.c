// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem -analyzer-experimental-checks -analyzer-checker=core.experimental.UnreachableCode -verify -analyzer-constraints=basic %s
// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem -analyzer-experimental-checks -analyzer-checker=core.experimental.UnreachableCode -verify -analyzer-constraints=range %s

// These are used to trigger warnings.
typedef typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
#define NULL ((void*)0)
#define UINT_MAX -1U

//---------------
//  Plus/minus
//---------------

void separateExpressions (int a) {
  int b = a + 1;
  --b;

  char* buf = malloc(1);
  if (a != 0 && b == 0)
    return; // expected-warning{{never executed}}
  free(buf);
}

void oneLongExpression (int a) {
  // Expression canonicalization should still allow this to work, even though
  // the first term is on the left.
  int b = 15 + a + 15 - 10 - 20;

  char* buf = malloc(1);
  if (a != 0 && b == 0)
    return; // expected-warning{{never executed}}
  free(buf);
}

void mixedTypes (int a) {
  char* buf = malloc(1);

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
  char* b = NULL;
  if (a == UINT_MAX)
    b = malloc(1);
  if (a+1 != 0)
    return; // no-warning
  if (a-1 != UINT_MAX-1)
    return; // no-warning
  free(b);
}

void ne_eq (unsigned a) {
  char* b = NULL;
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
  char* b = NULL;
  if (a == 1)
    b = malloc(1);
  if (a+1U != 2)
    return; // no-warning
  if (a-1U != 0)
    return; // expected-warning{{never executed}}
  free(b);
}

void mixed_ne_eq (int a) {
  char* b = NULL;
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
  char* b = NULL;
  if (a > 0)
    b = malloc(1);
  if (a == 0)
    return; // no-warning
  free(b);
}

void baselineGE (unsigned a) {
  char* b = NULL;
  if (a >= UINT_MAX)
    b = malloc(1);
  if (a == UINT_MAX)
    free(b);
  return; // no-warning
}

void baselineLT (unsigned a) {
  char* b = NULL;
  if (a < UINT_MAX)
    b = malloc(1);
  if (a == UINT_MAX)
    return; // no-warning
  free(b);
}

void baselineLE (unsigned a) {
  char* b = NULL;
  if (a <= 0)
    b = malloc(1);
  if (a == 0)
    free(b);
  return; // no-warning
}


// Adjustment gives each of these an extra solution!
void adjustedGT (unsigned a) {
  char* b = NULL;
  if (a-1 > UINT_MAX-1)
    b = malloc(1);
  return; // expected-warning{{leak}}
}

void adjustedGE (unsigned a) {
  char* b = NULL;
  if (a-1 >= UINT_MAX-1)
    b = malloc(1);
  if (a == UINT_MAX)
    free(b);
  return; // expected-warning{{leak}}
}

void adjustedLT (unsigned a) {
  char* b = NULL;
  if (a+1 < 1)
    b = malloc(1);
  return; // expected-warning{{leak}}
}

void adjustedLE (unsigned a) {
  char* b = NULL;
  if (a+1 <= 1)
    b = malloc(1);
  if (a == 0)
    free(b);
  return; // expected-warning{{leak}}
}


// Tautologies
void tautologyGT (unsigned a) {
  char* b = malloc(1);
  if (a > UINT_MAX)
    return; // no-warning
  free(b);
}

void tautologyGE (unsigned a) {
  char* b = malloc(1);
  if (a >= 0) // expected-warning{{always true}}
    free(b);
  return; // no-warning
}

void tautologyLT (unsigned a) {
  char* b = malloc(1);
  if (a < 0) // expected-warning{{always false}}
    return; // expected-warning{{never executed}}
  free(b);
}

void tautologyLE (unsigned a) {
  char* b = malloc(1);
  if (a <= UINT_MAX)
    free(b);
  return; // no-warning
}
