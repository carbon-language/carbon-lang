// RUN: %clangxx_msan -m64 -O0 %s -o %t && not %t >%t.out 2>&1
// FileCheck --check-prefix=CHECK-KEEP-GOING %s <%t.out
// RUN: %clangxx_msan -m64 -O0 %s -o %t && MSAN_OPTIONS=keep_going=0 not %t >%t.out 2>&1
// FileCheck %s <%t.out
// RUN: %clangxx_msan -m64 -O0 %s -o %t && MSAN_OPTIONS=keep_going=1 not %t >%t.out 2>&1
// FileCheck --check-prefix=CHECK-KEEP-GOING %s <%t.out

// RUN: %clangxx_msan -m64 -mllvm -msan-keep-going=1 -O0 %s -o %t && not %t >%t.out 2>&1
// FileCheck --check-prefix=CHECK-KEEP-GOING %s <%t.out
// RUN: %clangxx_msan -m64 -mllvm -msan-keep-going=1 -O0 %s -o %t && MSAN_OPTIONS=keep_going=0 not %t >%t.out 2>&1
// FileCheck %s <%t.out
// RUN: %clangxx_msan -m64 -mllvm -msan-keep-going=1 -O0 %s -o %t && MSAN_OPTIONS=keep_going=1 not %t >%t.out 2>&1
// FileCheck --check-prefix=CHECK-KEEP-GOING %s <%t.out

// Test how -mllvm -msan-keep-going and MSAN_OPTIONS=keep_going affect reports
// from interceptors.
// -mllvm -msan-keep-going provides the default value of keep_going flag, but is
// always overwritten by MSAN_OPTIONS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  char *volatile x = (char*)malloc(5 * sizeof(char));
  x[4] = 0;
  if (strlen(x) < 3)
    exit(0);
  fprintf(stderr, "Done\n");
  // CHECK-NOT: Done
  // CHECK-KEEP-GOING: Done
  return 0;
}
