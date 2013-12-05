// RUN: %clangxx_asan -O2 %s -o %t
// RUN: not %t -2 2>&1 | FileCheck --check-prefix=CHECK-m2 %s
// RUN: not %t -1 2>&1 | FileCheck --check-prefix=CHECK-m1 %s
// RUN: %t 0
// RUN: %t 8
// RUN: not %t 9  2>&1 | FileCheck --check-prefix=CHECK-9  %s
// RUN: not %t 10 2>&1 | FileCheck --check-prefix=CHECK-10 %s
// RUN: not %t 62 2>&1 | FileCheck --check-prefix=CHECK-62 %s
// RUN: not %t 63 2>&1 | FileCheck --check-prefix=CHECK-63 %s
// RUN: not %t 63 2>&1 | FileCheck --check-prefix=CHECK-63 %s
// RUN: not %t 73 2>&1 | FileCheck --check-prefix=CHECK-73 %s
// RUN: not %t 74 2>&1 | FileCheck --check-prefix=CHECK-74 %s
// RUN: not %t 126 2>&1 | FileCheck --check-prefix=CHECK-126 %s
// RUN: not %t 127 2>&1 | FileCheck --check-prefix=CHECK-127 %s
// RUN: not %t 137 2>&1 | FileCheck --check-prefix=CHECK-137 %s
// RUN: not %t 138 2>&1 | FileCheck --check-prefix=CHECK-138 %s
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
// CHECK-m2:  'AAA' <== Memory access at offset 30 underflows this variable
// CHECK-m1:  'AAA' <== Memory access at offset 31 partially underflows this variable
// CHECK-9:   'AAA' <== Memory access at offset 41 partially overflows this variable
// CHECK-10:  'AAA' <== Memory access at offset 42 overflows this variable
// CHECK-62:  'BBB' <== Memory access at offset 94 underflows this variable
// CHECK-63:  'BBB' <== Memory access at offset 95 partially underflows this variable
// CHECK-73:  'BBB' <== Memory access at offset 105 partially overflows this variable
// CHECK-74:  'BBB' <== Memory access at offset 106 overflows this variable
// CHECK-126: 'CCC' <== Memory access at offset 158 underflows this variable
// CHECK-127: 'CCC' <== Memory access at offset 159 partially underflows this variable
// CHECK-137: 'CCC' <== Memory access at offset 169 partially overflows this variable
// CHECK-138: 'CCC' <== Memory access at offset 170 overflows this variable
