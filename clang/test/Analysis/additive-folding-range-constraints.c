// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.core -verify -analyzer-constraints=range %s

// These are used to trigger warnings.
typedef typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
#define NULL ((void*)0)
#define UINT_MAX (__INT_MAX__  *2U +1U)

// Each of these adjusted ranges has an adjustment small enough to split the
// solution range across an overflow boundary (Min for <, Max for >).
// This corresponds to one set of branches in RangeConstraintManager.
void smallAdjustmentGT (unsigned a) {
  char* b = NULL;
  if (a+2 > 1)
    b = malloc(1);
  if (a == UINT_MAX-1 || a == UINT_MAX)
    return; // no-warning
  else if (a < UINT_MAX-1)
    free(b);
  return; // no-warning
}

void smallAdjustmentGE (unsigned a) {
  char* b = NULL;
  if (a+2 >= 1)
    b = malloc(1);
  if (a == UINT_MAX-1)
    return; // no-warning
  else if (a < UINT_MAX-1 || a == UINT_MAX)
    free(b);
  return; // no-warning
}

void smallAdjustmentLT (unsigned a) {
  char* b = NULL;
  if (a+1 < 2)
    b = malloc(1);
  if (a == 0 || a == UINT_MAX)
    free(b);
  return; // no-warning
}

void smallAdjustmentLE (unsigned a) {
  char* b = NULL;
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
  char* b = NULL;
  if (a-2 > UINT_MAX-1)
    b = malloc(1);
  if (a == 1 || a == 0)
    free(b);
  else if (a > 1)
    free(b);
  return; // no-warning
}

void largeAdjustmentGE (unsigned a) {
  char* b = NULL;
  if (a-2 >= UINT_MAX-1)
    b = malloc(1);
  if (a > 1)
    return; // no-warning
  else if (a == 1 || a == 0)
    free(b);
  return; // no-warning
}

void largeAdjustmentLT (unsigned a) {
  char* b = NULL;
  if (a+2 < 1)
    b = malloc(1);
  if (a == UINT_MAX-1 || a == UINT_MAX)
    free(b);
  else if (a < UINT_MAX-1)
    return; // no-warning
  return; // no-warning
}

void largeAdjustmentLE (unsigned a) {
  char* b = NULL;
  if (a+2 <= 1)
    b = malloc(1);
  if (a < UINT_MAX-1)
    return; // no-warning
  else if (a == UINT_MAX-1 || a == UINT_MAX)
    free(b);
  return; // no-warning
}
