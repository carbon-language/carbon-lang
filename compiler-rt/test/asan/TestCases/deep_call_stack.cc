// Check that UAR mode can handle very deep recusrion.
// export ASAN_OPTIONS=detect_stack_use_after_return=1
// RUN: %clangxx_asan -O2 %s -o %t && \
// RUN:   %t 2>&1 | FileCheck %s
// Also check that use_sigaltstack+verbosity doesn't crash.
// RUN: ASAN_OPTIONS=verbosity=1:use_sigaltstack=1 %t  | FileCheck %s
#include <stdio.h>

__attribute__((noinline))
void RecursiveFunc(int depth, int *ptr) {
  if ((depth % 1000) == 0)
    printf("[%05d] ptr: %p\n", depth, ptr);
  if (depth == 0)
    return;
  int local;
  RecursiveFunc(depth - 1, &local);
}

int main(int argc, char **argv) {
  RecursiveFunc(40000, 0);
  return 0;
}
// CHECK: [40000] ptr:
// CHECK: [20000] ptr:
// CHECK: [00000] ptr
