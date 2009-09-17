// RUN: clang-cc -emit-llvm %s -o %t -triple=x86_64-apple-darwin9 && 

struct X {};

// RUN: grep "define void @_Z1f1XS_" %t | count 1 &&
void f(X, X) { }

// RUN: grep "define void @_Z1fR1XS0_" %t | count 1 &&
void f(X&, X&) { }

// RUN: grep "define void @_Z1fRK1XS1_" %t | count 1 &&
void f(const X&, const X&) { }

typedef void T();
struct S {};

// RUN: grep "define void @_Z1fPFvvEM1SFvvE" %t | count 1 &&
void f(T*, T (S::*)) {}

// RUN: true
