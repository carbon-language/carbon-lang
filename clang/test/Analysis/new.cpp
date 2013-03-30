// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-store region -std=c++11 -verify %s
#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_eval(bool);

typedef __typeof__(sizeof(int)) size_t;
extern "C" void *malloc(size_t);
extern "C" void free(void *);

int someGlobal;
void testImplicitlyDeclaredGlobalNew() {
  if (someGlobal != 0)
    return;

  // This used to crash because the global operator new is being implicitly
  // declared and it does not have a valid source location. (PR13090)
  void *x = ::operator new(0);
  ::operator delete(x);

  // Check that the new/delete did not invalidate someGlobal;
  clang_analyzer_eval(someGlobal == 0); // expected-warning{{TRUE}}
}

void *testPlacementNew() {
  int *x = (int *)malloc(sizeof(int));
  *x = 1;
  clang_analyzer_eval(*x == 1); // expected-warning{{TRUE}};

  void *y = new (x) int;
  clang_analyzer_eval(x == y); // expected-warning{{TRUE}};
  clang_analyzer_eval(*x == 1); // expected-warning{{UNKNOWN}};

  return y;
}

void *operator new(size_t, size_t, int *);
void *testCustomNew() {
  int x[1] = {1};
  clang_analyzer_eval(*x == 1); // expected-warning{{TRUE}};

  void *y = new (0, x) int;
  clang_analyzer_eval(*x == 1); // expected-warning{{UNKNOWN}};

  return y; // no-warning
}

void *operator new(size_t, void *, void *);
void *testCustomNewMalloc() {
  int *x = (int *)malloc(sizeof(int));

  // Should be no-warning (the custom allocator could have freed x).
  void *y = new (0, x) int; // no-warning

  return y;
}

void testScalarInitialization() {
  int *n = new int(3);
  clang_analyzer_eval(*n == 3); // expected-warning{{TRUE}}

  new (n) int();
  clang_analyzer_eval(*n == 0); // expected-warning{{TRUE}}

  new (n) int{3};
  clang_analyzer_eval(*n == 3); // expected-warning{{TRUE}}

  new (n) int{};
  clang_analyzer_eval(*n == 0); // expected-warning{{TRUE}}
}

struct PtrWrapper {
  int *x;

  PtrWrapper(int *input) : x(input) {}
};

PtrWrapper *testNewInvalidation() {
  // Ensure that we don't consider this a leak.
  return new PtrWrapper(static_cast<int *>(malloc(4))); // no-warning
}

void testNewInvalidationPlacement(PtrWrapper *w) {
  // Ensure that we don't consider this a leak.
  new (w) PtrWrapper(static_cast<int *>(malloc(4))); // no-warning
}

int **testNewInvalidationScalar() {
  // Ensure that we don't consider this a leak.
  return new (int *)(static_cast<int *>(malloc(4))); // no-warning
}

void testNewInvalidationScalarPlacement(int **p) {
  // Ensure that we don't consider this a leak.
  new (p) (int *)(static_cast<int *>(malloc(4))); // no-warning
}

void testCacheOut(PtrWrapper w) {
  extern bool coin();
  if (coin())
    w.x = 0;
  new (&w.x) (int*)(0); // we cache out here; don't crash
}


//--------------------------------------------------------------------
// Check for intersection with other checkers from MallocChecker.cpp 
// bounded with unix.Malloc
//--------------------------------------------------------------------

// new/delete oparators are subjects of cplusplus.NewDelete.
void testNewDeleteNoWarn() {
  int i;
  delete &i; // no-warning

  int *p1 = new int;
  delete ++p1; // no-warning

  int *p2 = new int;
  delete p2;
  delete p2; // no-warning

  int *p3 = new int; // no-warning
}

// unix.Malloc does not know about operators new/delete.
void testDeleteMallocked() {
  int *x = (int *)malloc(sizeof(int));
  delete x; // FIXME: Shoud detect pointer escape and keep silent after 'delete' is modeled properly.
} // expected-warning{{Memory is never released; potential leak}}

void testDeleteOpAfterFree() {
  int *p = (int *)malloc(sizeof(int));
  free(p);
  operator delete(p); // expected-warning{{Use of memory after it is freed}}
}

void testDeleteAfterFree() {
  int *p = (int *)malloc(sizeof(int));
  free(p);
  delete p; // expected-warning{{Use of memory after it is freed}}
}

void testStandardPlacementNewAfterFree() {
  int *p = (int *)malloc(sizeof(int));
  free(p);
  p = new(p) int; // expected-warning{{Use of memory after it is freed}}
}

void testCustomPlacementNewAfterFree() {
  int *p = (int *)malloc(sizeof(int));
  free(p);
  p = new(0, p) int; // expected-warning{{Use of memory after it is freed}}
}

//--------------------------------
// Incorrectly-modelled behavior
//--------------------------------

int testNoInitialization() {
  int *n = new int;

  // Should warn that *n is uninitialized.
  if (*n) { // no-warning
    delete n;
    return 0;
  }
  delete n;
  return 1;
}

int testNoInitializationPlacement() {
  int n;
  new (&n) int;

  // Should warn that n is uninitialized.
  if (n) { // no-warning
    return 0;
  }
  return 1;
}
