// RUN: %clangxx_asan -O2 %s -o %t
// RUN: not %t -2 2>&1 | FileCheck --check-prefix=CHECK-m2 %s
// RUN: not %t -1 2>&1 | FileCheck --check-prefix=CHECK-m1 %s
// RUN: %t 0
// RUN: %t 8
// RUN: not %t 9  2>&1 | FileCheck --check-prefix=CHECK-9  %s
// RUN: not %t 10 2>&1 | FileCheck --check-prefix=CHECK-10 %s
// RUN: not %t 30 2>&1 | FileCheck --check-prefix=CHECK-30 %s
// RUN: not %t 31 2>&1 | FileCheck --check-prefix=CHECK-31 %s
// RUN: not %t 41 2>&1 | FileCheck --check-prefix=CHECK-41 %s
// RUN: not %t 42 2>&1 | FileCheck --check-prefix=CHECK-42 %s
// RUN: not %t 62 2>&1 | FileCheck --check-prefix=CHECK-62 %s
// RUN: not %t 63 2>&1 | FileCheck --check-prefix=CHECK-63 %s
// RUN: not %t 73 2>&1 | FileCheck --check-prefix=CHECK-73 %s
// RUN: not %t 74 2>&1 | FileCheck --check-prefix=CHECK-74 %s
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
int main(int argc, char **argv) {
  assert(argc >= 2);
  int idx = atoi(argv[1]);
  char AAA[10], BBB[10], CCC[10];
  memset(AAA, 0, sizeof(AAA));
  memset(BBB, 0, sizeof(BBB));
  memset(CCC, 0, sizeof(CCC));
  int res = 0;
  char *p = AAA + idx;
  printf("AAA: %p\ny: %p\nz: %p\np: %p\n", AAA, BBB, CCC, p);
  // make sure BBB and CCC are not removed;
  return *(short*)(p) + BBB[argc % 2] + CCC[argc % 2];
}
// CHECK-m2: 'AAA' <== {{.*}}underflows this variable
// CHECK-m1: 'AAA' <== {{.*}}partially underflows this variable
// CHECK-9:  'AAA' <== {{.*}}partially overflows this variable
// CHECK-10: 'AAA' <== {{.*}}overflows this variable
// CHECK-30: 'BBB' <== {{.*}}underflows this variable
// CHECK-31: 'BBB' <== {{.*}}partially underflows this variable
// CHECK-41: 'BBB' <== {{.*}}partially overflows this variable
// CHECK-42: 'BBB' <== {{.*}}overflows this variable
// CHECK-62: 'CCC' <== {{.*}}underflows this variable
// CHECK-63: 'CCC' <== {{.*}}partially underflows this variable
// CHECK-73: 'CCC' <== {{.*}}partially overflows this variable
// CHECK-74: 'CCC' <== {{.*}}overflows this variable
