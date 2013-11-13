// RUN: %clangxx_asan -O0 %s -o %t 2>&1
// RUN: not %t 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=MALLOC-CTX

// Also works if no malloc context is available.
// RUN: ASAN_OPTIONS=malloc_context_size=0:fast_unwind_on_malloc=0 not %t 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=malloc_context_size=0:fast_unwind_on_malloc=1 not %t 2>&1 | FileCheck %s

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char*)malloc(10 * sizeof(char));
  memset(x, 0, 10);
  int res = x[argc];
  free(x);
  free(x + argc - 1);  // BOOM
  // CHECK: AddressSanitizer: attempting double-free{{.*}}in thread T0
  // CHECK: double-free.cc:[[@LINE-2]]
  // CHECK: freed by thread T0 here:
  // MALLOC-CTX: double-free.cc:[[@LINE-5]]
  // CHECK: allocated by thread T0 here:
  // MALLOC-CTX: double-free.cc:[[@LINE-10]]
  return res;
}
