// RUN: %clangxx -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

// Malloc/free hooks are not supported on Windows.
// XFAIL: win32

#include <stdlib.h>
#include <unistd.h>
#include <sanitizer/allocator_interface.h>

extern "C" {
const volatile void *global_ptr;

// Note: avoid calling functions that allocate memory in malloc/free
// to avoid infinite recursion.
void __sanitizer_malloc_hook(const volatile void *ptr, size_t sz) {
  if (__sanitizer_get_ownership(ptr)) {
    write(1, "MallocHook\n", sizeof("MallocHook\n"));
    global_ptr = ptr;
  }
}
void __sanitizer_free_hook(const volatile void *ptr) {
  if (__sanitizer_get_ownership(ptr) && ptr == global_ptr)
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
