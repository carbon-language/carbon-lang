// RUN: %clangxx_asan -O2 %s -o %t
// RUN: %t 2>&1 | FileCheck %s
#include <stdio.h>
#include <stdlib.h>

extern "C" {
// Note: avoid calling functions that allocate memory in malloc/free
// to avoid infinite recursion.
void __asan_malloc_hook(void *ptr, size_t sz) { puts("MallocHook"); }
void __asan_free_hook(void *ptr) {  puts("FreeHook"); }
}  // extern "C"

int main() {
  volatile int *x = new int;
  // CHECK: MallocHook
  *x = 0;
  delete x;
  // CHECK: FreeHook
  return 0;
}
