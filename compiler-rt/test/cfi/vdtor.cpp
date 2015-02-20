// RUN: %clangxx_cfi -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx -o %t %s
// RUN: %t 2>&1 | FileCheck --check-prefix=NCFI %s

// Tests that the CFI enforcement also applies to virtual destructor calls made
// via 'delete'.

#include <stdio.h>

struct A {
  virtual ~A();
};

A::~A() {}

struct B {
  virtual ~B();
};

B::~B() {}

int main() {
  A *a = new A;

  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  delete (B *)a; // UB here

  // CFI-NOT: 2
  // NCFI: 2
  fprintf(stderr, "2\n");
}
