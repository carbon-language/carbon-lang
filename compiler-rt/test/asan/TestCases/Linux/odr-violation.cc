// Different size: detect a bug if detect_odr_violation>=1
// RUN: %clangxx_asan -DBUILD_SO=1 -fPIC -shared %s -o %t.so
// RUN: %clangxx_asan %s %t.so -Wl,-R. -o %t
// RUN: ASAN_OPTIONS=detect_odr_violation=1 not %run %t 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=detect_odr_violation=2 not %run %t 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=detect_odr_violation=0     %run %t 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN:                                         %run %t 2>&1 | FileCheck %s --check-prefix=DISABLED
//
// Same size: report a bug only if detect_odr_violation>=2.
// RUN: %clangxx_asan -DBUILD_SO=1 -fPIC -shared %s -o %t.so -DSZ=100
// RUN: ASAN_OPTIONS=detect_odr_violation=1     %run %t 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: ASAN_OPTIONS=detect_odr_violation=2 not %run %t 2>&1 | FileCheck %s

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
