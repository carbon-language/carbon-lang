// RUN: %clang_cc1 -analyze -analyzer-checker=core,cplusplus.NewDelete -analyzer-store region -std=c++11 -fblocks -verify %s
#include "Inputs/system-header-simulator-cxx.h"
#include "Inputs/system-header-simulator-objc.h"

typedef __typeof__(sizeof(int)) size_t;
extern "C" void *malloc(size_t);
extern "C" void free(void *);
int *global;

//------------------
// check for leaks
//------------------

void testGlobalExprNewBeforeOverload1() {
  int *p = new int;
} // expected-warning{{Memory is never released; potential leak}}

void testGlobalExprNewBeforeOverload2() {
  int *p = ::new int;
} // expected-warning{{Memory is never released; potential leak}}

void testGlobalOpNewBeforeOverload() {
  void *p = operator new(0);
} // expected-warning{{Memory is never released; potential leak}}

void testMemIsOnHeap() {
  int *p = new int;
  if (global != p)
    global = p;
} // expected-warning{{Memory is never released; potential leak}}
//FIXME: currently a memory region for 'new' is not a heap region, that lead to 
//false-positive 'memory leak' ('global != p' is not evaluated to true and 'p'
//does not escape)

void *operator new(std::size_t);
void *operator new(std::size_t, double d);
void *operator new[](std::size_t);
void *operator new[](std::size_t, double d);

void testExprPlacementNew() {
  int i;
  int *p1 = new(&i) int; // no warn - standard placement new

  int *p2 = new(1.0) int; // no warn - overloaded placement new

  int *p3 = new (std::nothrow) int;
} // expected-warning{{Memory is never released; potential leak}}

void testExprPlacementNewArray() {
  int i;
  int *p1 = new(&i) int[1]; // no warn - standard placement new[]

  int *p2 = new(1.0) int[1]; // no warn - overloaded placement new[]

  int *p3 = new (std::nothrow) int[1];
} // expected-warning{{Memory is never released; potential leak}}

void testCustomOpNew() {
  void *p = operator new(0); // no warn - call to a custom new
}

void testGlobalExprNew() {
  void *p = ::new int; // no warn - call to a custom new
}

void testCustomExprNew() {
  int *p = new int; // no warn - call to a custom new
}

void testGlobalExprNewArray() {
  void *p = ::new int[1]; // no warn - call to a custom new
}

void testOverloadedExprNewArray() {
  int *p = new int[1]; // no warn - call to a custom new
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
  delete p; // expected-warning{{Argument to free() was allocated by alloca(), not malloc()}}
}

void testDoubleDelete() {
  int *p = new int;
  delete p;
  delete p; // expected-warning{{Attempt to free released memory}}
}

void testExprDeleteArg() {
  int i;
  delete &i; // expected-warning{{Argument to free() is the address of the local variable 'i', which is not memory allocated by malloc()}}
} // FIXME: 'free()' -> 'delete'; 'malloc()' -> 'new'

void testExprDeleteArrArg() {
  int i;
  delete[] &i; // expected-warning{{Argument to free() is the address of the local variable 'i', which is not memory allocated by malloc()}}
} // FIXME: 'free()' -> 'delete[]'; 'malloc()' -> 'new[]'

void testAllocDeallocNames() {
  int *p = new(std::nothrow) int[1];
  delete[] (++p); // expected-warning{{Argument to free() is offset by 4 bytes from the start of memory allocated by malloc()}}
} // FIXME: 'free()' -> 'delete[]'; 'malloc()' -> 'new[]'

//----------------------------------------------------------------------------
// Check for intersections with unix.Malloc and unix.MallocWithAnnotations 
// checkers bounded with cplusplus.NewDelete.
//----------------------------------------------------------------------------

// malloc()/free() are subjects of unix.Malloc and unix.MallocWithAnnotations
void testMallocFreeNoWarn() {
  int i;
  free(&i); // no-warning

  int *p1 = (int *)malloc(sizeof(int));
  free(++p1); // no-warning

  int *p2 = (int *)malloc(sizeof(int));
  free(p2);
  free(p2); // no-warning

  int *p3 = (int *)malloc(sizeof(int)); // no-warning
}

void testFreeNewed() {
  int *p = new int;
  free(p); // pointer escaped, no-warning
}

void testObjcFreeNewed() {
  int *p = new int;
  NSData *nsdata = [NSData dataWithBytesNoCopy:p length:sizeof(int) freeWhenDone:1]; // pointer escaped, no-warning
}

void testFreeAfterDelete() {
  int *p = new int;  
  delete p;
  free(p); // expected-warning{{Use of memory after it is freed}}
}

void testStandardPlacementNewAfterDelete() {
  int *p = new int;  
  delete p;
  p = new(p) int; // expected-warning{{Use of memory after it is freed}}
}
