// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.deadcode.UnreachableCode,unix.Malloc -verify -analyzer-constraints=range %s

// These are used to trigger warnings.
typedef typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
#define NULL ((void*)0)
#define UINT_MAX (~0U)
#define INT_MAX (UINT_MAX & (UINT_MAX >> 1))
#define INT_MIN (-INT_MAX - 1)


// Each of these adjusted ranges has an adjustment small enough to split the
// solution range across an overflow boundary (Min for <, Max for >).
// This corresponds to one set of branches in RangeConstraintManager.
void smallAdjustmentGT (unsigned a) {
  void *b = NULL;
  if (a+2 > 1)
    b = malloc(1);
  if (a == UINT_MAX-1 || a == UINT_MAX)
    return; // no-warning
  else if (a < UINT_MAX-1)
    free(b);
  return; // no-warning
}

void smallAdjustmentGE (unsigned a) {
  void *b = NULL;
  if (a+2 >= 1)
    b = malloc(1);
  if (a == UINT_MAX-1)
    return; // no-warning
  else if (a < UINT_MAX-1 || a == UINT_MAX)
    free(b);
  return; // no-warning
}

void smallAdjustmentLT (unsigned a) {
  void *b = NULL;
  if (a+1 < 2)
    b = malloc(1);
  if (a == 0 || a == UINT_MAX)
    free(b);
  return; // no-warning
}

void smallAdjustmentLE (unsigned a) {
  void *b = NULL;
  if (a+1 <= 2)
    b = malloc(1);
  if (a == 0 || a == 1 || a == UINT_MAX)
    free(b);
  return; // no-warning
}


// Each of these adjusted ranges has an adjustment large enough to push the
// comparison value over an overflow boundary (Min for <, Max for >).
// This corresponds to one set of branches in RangeConstraintManager.
void largeAdjustmentGT (unsigned a) {
  void *b = NULL;
  if (a-2 > UINT_MAX-1)
    b = malloc(1);
  if (a == 1 || a == 0)
    free(b);
  else if (a > 1)
    free(b);
  return; // no-warning
}

void largeAdjustmentGE (unsigned a) {
  void *b = NULL;
  if (a-2 >= UINT_MAX-1)
    b = malloc(1);
  if (a > 1)
    return; // no-warning
  else if (a == 1 || a == 0)
    free(b);
  return; // no-warning
}

void largeAdjustmentLT (unsigned a) {
  void *b = NULL;
  if (a+2 < 1)
    b = malloc(1);
  if (a == UINT_MAX-1 || a == UINT_MAX)
    free(b);
  else if (a < UINT_MAX-1)
    return; // no-warning
  return; // no-warning
}

void largeAdjustmentLE (unsigned a) {
  void *b = NULL;
  if (a+2 <= 1)
    b = malloc(1);
  if (a < UINT_MAX-1)
    return; // no-warning
  else if (a == UINT_MAX-1 || a == UINT_MAX)
    free(b);
  return; // no-warning
}


// Test the nine cases in RangeConstraintManager's pinning logic.
void mixedComparisons1(signed char a) {
  // Case 1: The range is entirely below the symbol's range.
  int min = INT_MIN;

  if ((a - 2) < (min + 5LL))
    return; // expected-warning{{never executed}}

  if (a == 0)
    return; // no-warning
  if (a == 0x7F)
    return; // no-warning
  if (a == -0x80)
    return; // no-warning
  return; // no-warning
}

void mixedComparisons2(signed char a) {
  // Case 2: Only the lower end of the range is outside.
  if ((a - 5) < (-0x81LL)) {
    if (a == 0)
      return; // expected-warning{{never executed}}
    if (a == 0x7F)
      return; // expected-warning{{never executed}}
    if (a == -0x80)
      return; // no-warning    
    return; // no-warning
  } else {
    return; // no-warning
  }
}

void mixedComparisons3(signed char a) {
  // Case 3: The entire symbol range is covered.
  if ((a - 0x200) < -0x100LL) {
    if (a == 0)
      return; // no-warning
    if (a == 0x7F)
      return; // no-warning
    if (a == -0x80)
      return; // no-warning    
    return; // no-warning
  } else {
    return; // expected-warning{{never executed}}
  }
}

void mixedComparisons4(signed char a) {
  // Case 4: The range wraps around, but the lower wrap is out-of-range.
  if ((a - 5) > 0LL) {
    if (a == 0)
      return; // expected-warning{{never executed}}
    if (a == 0x7F)
      return; // no-warning
    if (a == -0x80)
      return; // expected-warning{{never executed}}
    return; // no-warning
  } else {
    return; // no-warning
  }
}

void mixedComparisons5(signed char a) {
  // Case 5a: The range is inside and does not wrap.
  if ((a + 5) == 0LL) {
    if (a == 0)
      return; // expected-warning{{never executed}}
    if (a == 0x7F)
      return; // expected-warning{{never executed}}
    if (a == -0x80)
      return; // expected-warning{{never executed}}
    return; // no-warning
  } else {
    return; // no-warning
  }
}

void mixedComparisons5Wrap(signed char a) {
  // Case 5b: The range is inside and does wrap.
  if ((a + 5) != 0LL) {
    if (a == 0)
      return; // no-warning
    if (a == 0x7F)
      return; // no-warning
    if (a == -0x80)
      return; // no-warning
    return; // no-warning
  } else {
    return; // no-warning
  }
}

void mixedComparisons6(signed char a) {
  // Case 6: Only the upper end of the range is outside.
  if ((a + 5) > 0x81LL) {
    if (a == 0)
      return; // expected-warning{{never executed}}
    if (a == 0x7F)
      return; // no-warning
    if (a == -0x80)
      return; // expected-warning{{never executed}}
    return; // no-warning
  } else {
    return; // no-warning
  }
}

void mixedComparisons7(signed char a) {
  // Case 7: The range wraps around but is entirely outside the symbol's range.
  int min = INT_MIN;

  if ((a + 2) < (min + 5LL))
    return; // expected-warning{{never executed}}

  if (a == 0)
    return; // no-warning
  if (a == 0x7F)
    return; // no-warning
  if (a == -0x80)
    return; // no-warning
  return; // no-warning
}

void mixedComparisons8(signed char a) {
  // Case 8: The range wraps, but the upper wrap is out of range.
  if ((a + 5) < 0LL) {
    if (a == 0)
      return; // expected-warning{{never executed}}
    if (a == 0x7F)
      return; // expected-warning{{never executed}}
    if (a == -0x80)
      return; // no-warning
    return; // no-warning
  } else {
    return; // no-warning
  }
}

void mixedComparisons9(signed char a) {
  // Case 9: The range is entirely above the symbol's range.
  int max = INT_MAX;

  if ((a + 2) > (max - 5LL))
    return; // expected-warning{{never executed}}

  if (a == 0)
    return; // no-warning
  if (a == 0x7F)
    return; // no-warning
  if (a == -0x80)
    return; // no-warning
  return; // no-warning
}
