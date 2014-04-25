// RUN: %clangxx_asan -DBUILD_SO=1 -fPIC -shared %s -o %t.so
// RUN: %clangxx_asan %s %t.so -Wl,-R. -o %t
// RUN: ASAN_OPTIONS=detect_odr_violation=1 not %t 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=detect_odr_violation=0     %t 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN:                                         %t 2>&1 | FileCheck %s --check-prefix=DISABLED

#ifndef SZ
# define SZ 4
#endif

#if BUILD_SO
char G[SZ];
#else
#include <stdio.h>
char G[100];
int main(int argc, char **argv) {
  printf("PASS: %p\n", &G);
}
#endif

// CHECK: ERROR: AddressSanitizer: odr-violation
// CHECK: size=100 G
// DISABLED: PASS
