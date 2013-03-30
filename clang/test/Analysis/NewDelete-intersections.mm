// RUN: %clang_cc1 -analyze -analyzer-checker=core,cplusplus.NewDelete -analyzer-store region -std=c++11 -fblocks -verify %s
#include "Inputs/system-header-simulator-cxx.h"
#include "Inputs/system-header-simulator-objc.h"

typedef __typeof__(sizeof(int)) size_t;
extern "C" void *malloc(size_t);
extern "C" void free(void *);

//----------------------------------------------------------------------------
// Check for intersections with unix.Malloc and unix.MallocWithAnnotations 
// checkers bounded with cplusplus.NewDelete.
//----------------------------------------------------------------------------

// malloc()/free() are subjects of unix.Malloc and unix.MallocWithAnnotations
void testMallocFreeNoWarn() {
  int i;
  free(&i); // no warn

  int *p1 = (int *)malloc(sizeof(int));
  free(++p1); // no warn

  int *p2 = (int *)malloc(sizeof(int));
  free(p2);
  free(p2); // no warn

  int *p3 = (int *)malloc(sizeof(int)); // no warn
}

void testDeleteMalloced() {
  int *p = (int *)malloc(sizeof(int));
  delete p; // no warn
} 

// FIXME: Pointer should escape

//----- Test free standard new
void testFreeOpNew() {
  void *p = operator new(0);
  free(p);
} // expected-warning{{Memory is never released; potential leak}}
// FIXME: Pointer should escape

void testFreeNewExpr() {
  int *p = new int;
  free(p);
} // expected-warning{{Memory is never released; potential leak}}
// FIXME: Pointer should escape

void testObjcFreeNewed() {
  int *p = new int;
  NSData *nsdata = [NSData dataWithBytesNoCopy:p length:sizeof(int) freeWhenDone:1]; // expected-warning{{Memory is never released; potential leak}}
}
// FIXME: Pointer should escape

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
