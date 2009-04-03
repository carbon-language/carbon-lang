// RUN: clang-cc -fsyntax-only -fixit-at=fixit-at.c:3:1 %s -o %t.m &&
// RUN: clang-cc -verify %t.m

@protocol X;

void foo() {
  <X> *P;    // should be fixed to 'id<X>'.
}
