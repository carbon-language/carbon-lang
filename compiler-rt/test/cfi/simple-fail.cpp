// RUN: %clangxx_cfi -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx -o %t %s
// RUN: %t 2>&1 | FileCheck --check-prefix=NCFI %s

// Tests that the CFI mechanism crashes the program when making a virtual call
// to an object of the wrong class but with a compatible vtable, by casting a
// pointer to such an object and attempting to make a call through it.

#include <stdio.h>

struct A {
  virtual void f();
};

void A::f() {}

struct B {
  virtual void f();
};

void B::f() {}

int main() {
  A *a = new A;

  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  ((B *)a)->f(); // UB here

  // CFI-NOT: 2
  // NCFI: 2
  fprintf(stderr, "2\n");
}
