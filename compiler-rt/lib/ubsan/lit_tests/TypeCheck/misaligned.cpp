// RUN: %clang -fsanitize=alignment %s -O3 -o %t
// RUN: %t l0 && %t s0 && %t r0 && %t m0 && %t f0 && %t n0
// RUN: %t l1 2>&1 | FileCheck %s --check-prefix=CHECK-LOAD
// RUN: %t s1 2>&1 | FileCheck %s --check-prefix=CHECK-STORE
// RUN: %t r1 2>&1 | FileCheck %s --check-prefix=CHECK-REFERENCE
// RUN: %t m1 2>&1 | FileCheck %s --check-prefix=CHECK-MEMBER
// RUN: %t f1 2>&1 | FileCheck %s --check-prefix=CHECK-MEMFUN
// RUN: %t n1 2>&1 | FileCheck %s --check-prefix=CHECK-NEW

#include <new>

struct S {
  S() {}
  int f() { return 0; }
  int k;
};

int main(int, char **argv) {
  char c[5] __attribute__((aligned(4))) = {};

  // Pointer value may be unspecified here, but behavior is not undefined.
  int *p = (int*)&c[argv[1][1] - '0'];
  S *s = (S*)p;

  (void)*p; // ok!

  switch (argv[1][0]) {
  case 'l':
    // CHECK-LOAD: misaligned.cpp:[[@LINE+1]]:12: runtime error: load of misaligned address 0x{{[0-9a-f]*}} for type 'int', which requires 4 byte alignment
    return *p;
  case 's':
    // CHECK-STORE: misaligned.cpp:[[@LINE+1]]:5: runtime error: store to misaligned address 0x{{[0-9a-f]*}} for type 'int', which requires 4 byte alignment
    *p = 1;
    break;
  case 'r':
    // CHECK-REFERENCE: misaligned.cpp:[[@LINE+1]]:15: runtime error: reference binding to misaligned address 0x{{[0-9a-f]*}} for type 'int', which requires 4 byte alignment
    {int &r = *p;}
    break;
  case 'm':
    // CHECK-MEMBER: misaligned.cpp:[[@LINE+1]]:15: runtime error: member access within misaligned address 0x{{[0-9a-f]*}} for type 'S', which requires 4 byte alignment
    return s->k;
  case 'f':
    // CHECK-MEMFUN: misaligned.cpp:[[@LINE+1]]:12: runtime error: member call on misaligned address 0x{{[0-9a-f]*}} for type 'S', which requires 4 byte alignment
    return s->f();
  case 'n':
    // FIXME: Provide a better source location here.
    // CHECK-NEW: misaligned{{.*}}:0x{{[0-9a-f]*}}: runtime error: constructor call on misaligned address 0x{{[0-9a-f]*}} for type 'S', which requires 4 byte alignment
    return (new (s) S)->k;
  }
}
