// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-config suppress-null-return-paths=false -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core -verify -DSUPPRESSED %s

int opaquePropertyCheck(void *object);
int coin();

int *dynCastToInt(void *ptr) {
  if (opaquePropertyCheck(ptr))
    return (int *)ptr;
  return 0;
}

int *dynCastOrNull(void *ptr) {
  if (!ptr)
    return 0;
  if (opaquePropertyCheck(ptr))
    return (int *)ptr;
  return 0;
}


void testDynCast(void *p) {
  int *casted = dynCastToInt(p);
  *casted = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void testDynCastOrNull(void *p) {
  int *casted = dynCastOrNull(p);
  *casted = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}


void testBranch(void *p) {
  int *casted;

  // Although the report will be suppressed on one branch, it should still be
  // valid on the other.
  if (coin()) {
    casted = dynCastToInt(p);
  } else {
    if (p)
      return;
    casted = (int *)p;
  }

  *casted = 1; // expected-warning {{Dereference of null pointer}}
}

void testBranchReversed(void *p) {
  int *casted;

  // Although the report will be suppressed on one branch, it should still be
  // valid on the other.
  if (coin()) {
    if (p)
      return;
    casted = (int *)p;
  } else {
    casted = dynCastToInt(p);
  }

  *casted = 1; // expected-warning {{Dereference of null pointer}}
}


// ---------------------------------------
// FALSE NEGATIVES (over-suppression)
// ---------------------------------------

void testDynCastOrNullOfNull() {
  // In this case we have a known value for the argument, and thus the path
  // through the function doesn't ever split.
  int *casted = dynCastOrNull(0);
  *casted = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void testDynCastOfNull() {
  // In this case all paths out of the function return 0, but they are all
  // dominated by a branch whose condition we don't know!
  int *casted = dynCastToInt(0);
  *casted = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

