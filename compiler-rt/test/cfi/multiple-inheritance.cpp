// RUN: %clangxx_cfi -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: not --crash %t x 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx -o %t %s
// RUN: %t 2>&1 | FileCheck --check-prefix=NCFI %s
// RUN: %t x 2>&1 | FileCheck --check-prefix=NCFI %s

// Tests that the CFI mechanism is sensitive to multiple inheritance and only
// permits calls via virtual tables for the correct base class.

#include <stdio.h>

struct A {
  virtual void f() = 0;
};

struct B {
  virtual void g() = 0;
};

struct C : A, B {
  virtual void f(), g();
};

void C::f() {}
void C::g() {}

int main(int argc, char **argv) {
  C *c = new C;

  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  if (argc > 1) {
    A *a = c;
    ((B *)a)->g(); // UB here
  } else {
    B *b = c;
    ((A *)b)->f(); // UB here
  }

  // CFI-NOT: 2
  // NCFI: 2
  fprintf(stderr, "2\n");
}
