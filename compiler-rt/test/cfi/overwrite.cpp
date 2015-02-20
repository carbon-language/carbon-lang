// RUN: %clangxx_cfi -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx -o %t %s
// RUN: %t 2>&1 | FileCheck --check-prefix=NCFI %s

// Tests that the CFI mechanism crashes the program when a virtual table is
// replaced with a compatible table of function pointers that does not belong to
// any class, by manually overwriting the virtual table of an object and
// attempting to make a call through it.

#include <stdio.h>

struct A {
  virtual void f();
};

void A::f() {}

void foo() {
  fprintf(stderr, "foo\n");
}

void *fake_vtable[] = { (void *)&foo };

int main() {
  A *a = new A;
  *((void **)a) = fake_vtable; // UB here

  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  // CFI-NOT: foo
  // NCFI: foo
  a->f();

  // CFI-NOT: 2
  // NCFI: 2
  fprintf(stderr, "2\n");
}
