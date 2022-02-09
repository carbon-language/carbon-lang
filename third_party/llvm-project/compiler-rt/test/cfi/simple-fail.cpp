// RUN: %clangxx_cfi -o %t1 %s
// RUN: %expect_crash %run %t1 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB32 -o %t2 %s
// RUN: %expect_crash %run %t2 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DB64 -o %t3 %s
// RUN: %expect_crash %run %t3 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -DBM -o %t4 %s
// RUN: %expect_crash %run %t4 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O1 -o %t5 %s
// RUN: %expect_crash %run %t5 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O1 -DB32 -o %t6 %s
// RUN: %expect_crash %run %t6 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O1 -DB64 -o %t7 %s
// RUN: %expect_crash %run %t7 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O1 -DBM -o %t8 %s
// RUN: %expect_crash %run %t8 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O2 -o %t9 %s
// RUN: %expect_crash %run %t9 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O2 -DB32 -o %t10 %s
// RUN: %expect_crash %run %t10 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O2 -DB64 -o %t11 %s
// RUN: %expect_crash %run %t11 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O2 -DBM -o %t12 %s
// RUN: %expect_crash %run %t12 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O3 -o %t13 %s
// RUN: %expect_crash %run %t13 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O3 -DB32 -o %t14 %s
// RUN: %expect_crash %run %t14 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O3 -DB64 -o %t15 %s
// RUN: %expect_crash %run %t15 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -O3 -DBM -o %t16 %s
// RUN: %expect_crash %run %t16 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi_diag -o %t17 %s
// RUN: %run %t17 2>&1 | FileCheck --check-prefix=CFI-DIAG %s

// RUN: %clangxx -o %t18 %s
// RUN: %run %t18 2>&1 | FileCheck --check-prefix=NCFI %s

// RUN: %clangxx_cfi -DCHECK_NO_SANITIZE_CFI -o %t19 %s
// RUN: %run %t19 2>&1 | FileCheck --check-prefix=NCFI %s

// Tests that the CFI mechanism crashes the program when making a virtual call
// to an object of the wrong class but with a compatible vtable, by casting a
// pointer to such an object and attempting to make a call through it.

// REQUIRES: cxxabi

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

#if defined(CHECK_NO_SANITIZE_CFI)
__attribute__((no_sanitize("cfi")))
#endif
int main() {
  create_derivers<B>();

  A *a = new A;
  break_optimization(a);

  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  // CFI-DIAG: runtime error: control flow integrity check for type 'B' failed during cast to unrelated type
  // CFI-DIAG-NEXT: note: vtable is of type '{{(struct )?}}A'
  // CFI-DIAG: runtime error: control flow integrity check for type 'B' failed during virtual call
  // CFI-DIAG-NEXT: note: vtable is of type '{{(struct )?}}A'
  ((B *)a)->f(); // UB here

  // CFI-NOT: {{^2$}}
  // NCFI: {{^2$}}
  fprintf(stderr, "2\n");
}
