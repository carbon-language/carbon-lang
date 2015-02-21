// RUN: %clangxx_cfi -c -DTU1 -o %t1.o %s
// RUN: %clangxx_cfi -c -DTU2 -o %t2.o %S/../cfi/anon-namespace.cpp
// RUN: %clangxx_cfi -o %t %t1.o %t2.o
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -c -DTU1 -DB32 -o %t1.o %s
// RUN: %clangxx_cfi -c -DTU2 -DB32 -o %t2.o %S/../cfi/anon-namespace.cpp
// RUN: %clangxx_cfi -o %t %t1.o %t2.o
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -c -DTU1 -DB64 -o %t1.o %s
// RUN: %clangxx_cfi -c -DTU2 -DB64 -o %t2.o %S/../cfi/anon-namespace.cpp
// RUN: %clangxx_cfi -o %t %t1.o %t2.o
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi -c -DTU1 -DBM -o %t1.o %s
// RUN: %clangxx_cfi -c -DTU2 -DBM -o %t2.o %S/../cfi/anon-namespace.cpp
// RUN: %clangxx_cfi -o %t %t1.o %t2.o
// RUN: not --crash %t 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx -c -DTU1 -o %t1.o %s
// RUN: %clangxx -c -DTU2 -o %t2.o %S/../cfi/anon-namespace.cpp
// RUN: %clangxx -o %t %t1.o %t2.o
// RUN: %t 2>&1 | FileCheck --check-prefix=NCFI %s

// Tests that the CFI mechanism treats classes in the anonymous namespace in
// different translation units as having distinct identities. This is done by
// compiling two translation units TU1 and TU2 containing a class named B in an
// anonymous namespace, and testing that the program crashes if TU2 attempts to
// use a TU1 B as a TU2 B.

// FIXME: This test should not require that the paths supplied to the compiler
// are different. It currently does so because bitset names have global scope
// so we have to mangle the file path into the bitset name.

#include <stdio.h>
#include "utils.h"

struct A {
  virtual void f() = 0;
};

namespace {

struct B : A {
  virtual void f() {}
};

}

A *mkb();

#ifdef TU1

A *mkb() {
  return new B;
}

#endif  // TU1

#ifdef TU2

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

  A *a = mkb();
  break_optimization(a);

  // CFI: 1
  // NCFI: 1
  fprintf(stderr, "1\n");

  ((B *)a)->f(); // UB here

  // CFI-NOT: 2
  // NCFI: 2
  fprintf(stderr, "2\n");
}

#endif  // TU2
