// RUN: %clangxx_msan -std=c++11 -O0 %s -o %t && %run %t
// RUN: %clangxx_msan -std=c++11 -O0 %s -o %t -DPOSITIVE && not %run %t |& FileCheck %s

#include <assert.h>
#include <sanitizer/msan_interface.h>
#include <string.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

int main(void) {
  char a[100];
  memset(a, 0xab, 100);

  char b[100];
  iovec iov_a[] = {{(void *)a, 20}, (void *)(a + 50), 10};
  iovec iov_b[] = {{(void *)(b + 10), 10}, (void *)(b + 30), 20};

  __msan_poison(&b, sizeof(b));
  ssize_t res = process_vm_readv(getpid(), iov_b, 2, iov_a, 2, 0);
  assert(res == 30);
  __msan_check_mem_is_initialized(b + 10, 10);
  __msan_check_mem_is_initialized(b + 30, 20);
  assert(__msan_test_shadow(b + 9, 1) == 0);
  assert(__msan_test_shadow(b + 20, 1) == 0);
  assert(__msan_test_shadow(b + 29, 1) == 0);
  assert(__msan_test_shadow(b + 50, 1) == 0);

#ifdef POSITIVE
  __msan_unpoison(&b, sizeof(b));
  __msan_poison(b + 32, 1);
  res = process_vm_writev(getpid(), iov_b, 2, iov_a, 2, 0);
// CHECK: Uninitialized bytes {{.*}} at offset 2 inside
// CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
// CHECK: #0 0x{{.*}} in {{.*}}process_vm_writev
#else
  __msan_unpoison(&b, sizeof(b));
  res = process_vm_writev(getpid(), iov_b, 2, iov_a, 2, 0);
  assert(res == 30);
#endif

  return 0;
}
