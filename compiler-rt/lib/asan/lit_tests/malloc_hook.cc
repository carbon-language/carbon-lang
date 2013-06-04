// RUN: %clangxx_asan -O2 %s -o %t
// RUN: %t 2>&1 | FileCheck %s
#include <stdlib.h>
#include <unistd.h>

extern "C" {
bool __asan_get_ownership(const void *p);

void *global_ptr;

// Note: avoid calling functions that allocate memory in malloc/free
// to avoid infinite recursion.
void __asan_malloc_hook(void *ptr, size_t sz) {
  if (__asan_get_ownership(ptr)) {
    write(1, "MallocHook\n", sizeof("MallocHook\n"));
    global_ptr = ptr;
  }
}
void __asan_free_hook(void *ptr) {
  if (__asan_get_ownership(ptr) && ptr == global_ptr)
    write(1, "FreeHook\n", sizeof("FreeHook\n"));
}
}  // extern "C"

int main() {
  volatile int *x = new int;
  // CHECK: MallocHook
  // Check that malloc hook was called with correct argument.
  if (global_ptr != (void*)x) {
    _exit(1);
  }
  *x = 0;
  delete x;
  // CHECK: FreeHook
  return 0;
}
