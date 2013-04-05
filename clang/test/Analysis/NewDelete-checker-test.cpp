// RUN: %clang_cc1 -analyze -analyzer-checker=core,cplusplus.NewDelete -std=c++11 -fblocks -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,cplusplus.NewDelete,alpha.cplusplus.NewDeleteLeaks -DLEAKS -std=c++11 -fblocks -verify %s
#include "Inputs/system-header-simulator-cxx.h"

typedef __typeof__(sizeof(int)) size_t;
extern "C" void *malloc(size_t);
int *global;

//------------------
// check for leaks
//------------------

//----- Standard non-placement operators
void testGlobalOpNew() {
  void *p = operator new(0);
}
#ifdef LEAKS
// expected-warning@-2{{Memory is never released; potential leak}}
#endif

void testGlobalOpNewArray() {
  void *p = operator new[](0);
}
#ifdef LEAKS
// expected-warning@-2{{Memory is never released; potential leak}}
#endif

void testGlobalNewExpr() {
  int *p = new int;
}
#ifdef LEAKS
// expected-warning@-2{{Memory is never released; potential leak}}
#endif

void testGlobalNewExprArray() {
  int *p = new int[0];
}
#ifdef LEAKS
// expected-warning@-2{{Memory is never released; potential leak}}
#endif

//----- Standard nothrow placement operators
void testGlobalNoThrowPlacementOpNewBeforeOverload() {
  void *p = operator new(0, std::nothrow);
}
#ifdef LEAKS
// expected-warning@-2{{Memory is never released; potential leak}}
#endif

void testGlobalNoThrowPlacementExprNewBeforeOverload() {
  int *p = new(std::nothrow) int;
}
#ifdef LEAKS
// expected-warning@-2{{Memory is never released; potential leak}}
#endif


//----- Standard pointer placement operators
void testGlobalPointerPlacementNew() {
  int i;

  void *p1 = operator new(0, &i); // no warn

  void *p2 = operator new[](0, &i); // no warn

  int *p3 = new(&i) int; // no warn

  int *p4 = new(&i) int[0]; // no warn
}

//----- Other cases
void testNewMemoryIsInHeap() {
  int *p = new int;
  if (global != p) // condition is always true as 'p' wraps a heap region that 
                   // is different from a region wrapped by 'global'
    global = p; // pointer escapes
}

struct PtrWrapper {
  int *x;

  PtrWrapper(int *input) : x(input) {}
};

void testNewInvalidationPlacement(PtrWrapper *w) {
  // Ensure that we don't consider this a leak.
  new (w) PtrWrapper(new int); // no warn
}

//---------------
// other checks
//---------------

void f(int *);

void testUseAfterDelete() {
  int *p = new int;
  delete p;
  f(p); // expected-warning{{Use of memory after it is freed}}
}

void testDeleteAlloca() {
  int *p = (int *)__builtin_alloca(sizeof(int));
  delete p; // expected-warning{{Memory allocated by alloca() should not be deallocated}}
}

void testDoubleDelete() {
  int *p = new int;
  delete p;
  delete p; // expected-warning{{Attempt to free released memory}}
}

void testExprDeleteArg() {
  int i;
  delete &i; // expected-warning{{Argument to 'delete' is the address of the local variable 'i', which is not memory allocated by 'new'}}
}

void testExprDeleteArrArg() {
  int i;
  delete[] &i; // expected-warning{{Argument to 'delete[]' is the address of the local variable 'i', which is not memory allocated by 'new[]'}}
}

void testAllocDeallocNames() {
  int *p = new(std::nothrow) int[1];
  delete[] (++p); // expected-warning{{Argument to 'delete[]' is offset by 4 bytes from the start of memory allocated by 'new[]'}}
}

//--------------------------------
// Test escape of newed const pointer. Note, a const pointer can be deleted.
//--------------------------------
struct StWithConstPtr {
  const int *memp;
};
void escape(const int &x);
void escapeStruct(const StWithConstPtr &x);
void escapePtr(const StWithConstPtr *x);
void escapeVoidPtr(const void *x);

void testConstEscape() {
  int *p = new int(1);
  escape(*p);
} // no-warning

void testConstEscapeStruct() {
  StWithConstPtr *St = new StWithConstPtr();
  escapeStruct(*St);
} // no-warning

void testConstEscapeStructPtr() {
  StWithConstPtr *St = new StWithConstPtr();
  escapePtr(St);
} // no-warning

void testConstEscapeMember() {
  StWithConstPtr St;
  St.memp = new int(2);
  escapeVoidPtr(St.memp);
} // no-warning

void testConstEscapePlacementNew() {
  int *x = (int *)malloc(sizeof(int));
  void *y = new (x) int;
  escapeVoidPtr(y);
} // no-warning
