// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.cplusplus.NewDelete,unix.Malloc -analyzer-store region -std=c++11 -fblocks -verify %s
#include "Inputs/system-header-simulator-cxx.h"

void *allocator(std::size_t size);

void *operator new[](std::size_t size) throw() { return allocator(size); }
void *operator new(std::size_t size) throw() { return allocator(size); }
void *operator new(std::size_t size, std::nothrow_t& nothrow) throw() { return allocator(size); }
void *operator new(std::size_t, double d);

class C {
public:
  void *operator new(std::size_t);  
};

void testNewMethod() {
  void *p1 = C::operator new(0); // no warn

  C *p2 = new C; // no warn

  C *c3 = ::new C;
} // expected-warning{{Memory is never released; potential leak}}

void testOpNewArray() {
  void *p = operator new[](0); // call is inlined, no warn
}

void testNewExprArray() {
  int *p = new int[0];
} // expected-warning{{Memory is never released; potential leak}}

//----- Custom non-placement operators
void testOpNew() {
  void *p = operator new(0); // call is inlined, no warn
}

void testNewExpr() {
  int *p = new int;
} // expected-warning{{Memory is never released; potential leak}}

//----- Custom NoThrow placement operators
void testOpNewNoThrow() {
  void *p = operator new(0, std::nothrow);
} // expected-warning{{Memory is never released; potential leak}}

void testNewExprNoThrow() {
  int *p = new(std::nothrow) int;
} // expected-warning{{Memory is never released; potential leak}}

//----- Custom placement operators
void testOpNewPlacement() {
  void *p = operator new(0, 0.1); // no warn
} 

void testNewExprPlacement() {
  int *p = new(0.1) int; // no warn
}
