// RUN: %clangxx_cfi_dso -DSHARED_LIB %s -fPIC -shared -o %t1-so.so
// RUN: %clangxx_cfi_dso %s -o %t1 %t1-so.so
// RUN: %expect_crash %t1 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %t1 x 2>&1 | FileCheck --check-prefix=CFI-CAST %s

// RUN: %clangxx_cfi_dso -DB32 -DSHARED_LIB %s -fPIC -shared -o %t2-so.so
// RUN: %clangxx_cfi_dso -DB32 %s -o %t2 %t2-so.so
// RUN: %expect_crash %t2 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %t2 x 2>&1 | FileCheck --check-prefix=CFI-CAST %s

// RUN: %clangxx_cfi_dso -DB64 -DSHARED_LIB %s -fPIC -shared -o %t3-so.so
// RUN: %clangxx_cfi_dso -DB64 %s -o %t3 %t3-so.so
// RUN: %expect_crash %t3 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %t3 x 2>&1 | FileCheck --check-prefix=CFI-CAST %s

// RUN: %clangxx_cfi_dso -DBM -DSHARED_LIB %s -fPIC -shared -o %t4-so.so
// RUN: %clangxx_cfi_dso -DBM %s -o %t4 %t4-so.so
// RUN: %expect_crash %t4 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %t4 x 2>&1 | FileCheck --check-prefix=CFI-CAST %s

// RUN: %clangxx -DBM -DSHARED_LIB %s -fPIC -shared -o %t5-so.so
// RUN: %clangxx -DBM %s -o %t5 %t5-so.so
// RUN: %t5 2>&1 | FileCheck --check-prefix=NCFI %s
// RUN: %t5 x 2>&1 | FileCheck --check-prefix=NCFI %s

// RUN: %clangxx -DBM -DSHARED_LIB %s -fPIC -shared -o %t6-so.so
// RUN: %clangxx_cfi_dso -DBM %s -o %t6 %t6-so.so
// RUN: %t6 2>&1 | FileCheck --check-prefix=NCFI %s
// RUN: %t6 x 2>&1 | FileCheck --check-prefix=NCFI %s

// Tests that the CFI mechanism crashes the program when making a virtual call
// to an object of the wrong class but with a compatible vtable, by casting a
// pointer to such an object and attempting to make a call through it.

// REQUIRES: cxxabi

#include <stdio.h>
#include <string.h>

struct A {
  virtual void f();
};

void *create_B();

#ifdef SHARED_LIB

#include "../utils.h"
struct B {
  virtual void f();
};
void B::f() {}

void *create_B() {
  create_derivers<B>();
  return (void *)(new B());
}

#else

void A::f() {}

int main(int argc, char *argv[]) {
  void *p = create_B();
  A *a;

  // CFI: =0=
  // CFI-CAST: =0=
  // NCFI: =0=
  fprintf(stderr, "=0=\n");

  if (argc > 1 && argv[1][0] == 'x') {
    // Test cast. BOOM.
    a = (A*)p;
  } else {
    // Invisible to CFI. Test virtual call later.
    memcpy(&a, &p, sizeof(a));
  }

  // CFI: =1=
  // CFI-CAST-NOT: =1=
  // NCFI: =1=
  fprintf(stderr, "=1=\n");

  a->f(); // UB here

  // CFI-NOT: =2=
  // CFI-CAST-NOT: =2=
  // NCFI: =2=
  fprintf(stderr, "=2=\n");
}
#endif
