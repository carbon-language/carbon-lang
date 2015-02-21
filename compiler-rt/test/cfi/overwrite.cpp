// RUN: %clangxx_cfi -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB32 -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB64 -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DBM -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx -o %t %s
// RUN: %t 2>&1 | FileCheck --check-prefix=NCFI %s

// Tests that the CFI mechanism crashes the program when a virtual table is
// replaced with a compatible table of function pointers that does not belong to
// any class, by manually overwriting the virtual table of an object and
// attempting to make a call through it.

#include <stdio.h>
#include "utils.h"

struct A {
  virtual void f();
};

void A::f() {}

void foo() {
  fprintf(stderr, "foo\n");
}

void *fake_vtable[] = { (void *)&foo };

int main() {
#ifdef B32
  break_optimization(new Deriver<A, 0>);
#endif

#ifdef B64
  break_optimization(new Deriver<A, 0>);
  break_optimization(new Deriver<A, 1>);
#endif

#ifdef BM
  break_optimization(new Deriver<A, 0>);
  break_optimization(new Deriver<A, 1>);
  break_optimization(new Deriver<A, 2>);
#endif

  A *a = new A;
  *((void **)a) = fake_vtable; // UB here
  break_optimization(a);

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
