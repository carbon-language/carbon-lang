// RUN: %clangxx_msan -O2 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s
#include <stdlib.h>
#include <unistd.h>

extern "C" {
int __msan_get_ownership(const void *p);

void *global_ptr;

// Note: avoid calling functions that allocate memory in malloc/free
// to avoid infinite recursion.
void __msan_malloc_hook(void *ptr, size_t sz) {
  if (__msan_get_ownership(ptr)) {
    write(1, "MallocHook\n", sizeof("MallocHook\n"));
    global_ptr = ptr;
  }
}
void __msan_free_hook(void *ptr) {
  if (__msan_get_ownership(ptr) && ptr == global_ptr)
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
