// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.cplusplus.NewDeleteLeaks -std=c++11 -fblocks -verify %s
#include "Inputs/system-header-simulator-cxx.h"

//----- Standard non-placement operators
void testGlobalOpNew() {
  void *p = operator new(0);
} // expected-warning{{Potential leak of memory pointed to by 'p'}}

void testGlobalOpNewArray() {
  void *p = operator new[](0);
} // expected-warning{{Potential leak of memory pointed to by 'p'}}

void testGlobalNewExpr() {
  int *p = new int;
} // expected-warning{{Potential leak of memory pointed to by 'p'}}

void testGlobalNewExprArray() {
  int *p = new int[0];
} // expected-warning{{Potential leak of memory pointed to by 'p'}}

//----- Standard nothrow placement operators
void testGlobalNoThrowPlacementOpNewBeforeOverload() {
  void *p = operator new(0, std::nothrow);
} // expected-warning{{Potential leak of memory pointed to by 'p'}}

void testGlobalNoThrowPlacementExprNewBeforeOverload() {
  int *p = new(std::nothrow) int;
} // expected-warning{{Potential leak of memory pointed to by 'p'}}
