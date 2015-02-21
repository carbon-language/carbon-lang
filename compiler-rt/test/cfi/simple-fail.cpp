// RUN: %clangxx_cfi -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB32 -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB64 -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DBM -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O1 -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O1 -DB32 -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O1 -DB64 -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O1 -DBM -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O2 -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O2 -DB32 -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O2 -DB64 -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O2 -DBM -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O3 -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O3 -DB32 -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O3 -DB64 -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O3 -DBM -o %t %s
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx -o %t %s
// RUN: %t 2>&1 | FileCheck --check-prefix=NCFI %s

// Tests that the CFI mechanism crashes the program when making a virtual call
// to an object of the wrong class but with a compatible vtable, by casting a
// pointer to such an object and attempting to make a call through it.

#include <stdio.h>
#include "utils.h"

struct A {
  virtual void f();
};

void A::f() {}

struct B {
  virtual void f();
};

void B::f() {}

int main() {
#ifdef B32
  break_optimization(new Deriver<B, 0>);
#endif

#ifdef B64
  break_optimization(new Deriver<B, 0>);
  break_optimization(new Deriver<B, 1>);
#endif

#ifdef BM
  break_optimization(new Deriver<B, 0>);
  break_optimization(new Deriver<B, 1>);
  break_optimization(new Deriver<B, 2>);
#endif

  A *a = new A;
  break_optimization(a);

  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  ((B *)a)->f(); // UB here

  // CFI-NOT: 2
  // NCFI: 2
  fprintf(stderr, "2\n");
}
