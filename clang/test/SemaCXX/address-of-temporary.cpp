// RUN: %clang_cc1 -fsyntax-only -Wno-error=address-of-temporary -verify %s
struct X { 
  X();
  X(int);
  X(int, int);
};

void f0() { (void)&X(); } // expected-warning{{taking the address of a temporary object}}
void f1() { (void)&X(1); } // expected-warning{{taking the address of a temporary object}}
void f2() { (void)&X(1, 2); } // expected-warning{{taking the address of a temporary object}}
void f3() { (void)&(X)1; } // expected-warning{{taking the address of a temporary object}}

