// Check that UAR mode can handle very deep recusrion.
//
// RUN: %clangxx_asan -fsanitize=use-after-return -O2 %s -o %t && \
// RUN:   %t 2>&1 | FileCheck %s
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
