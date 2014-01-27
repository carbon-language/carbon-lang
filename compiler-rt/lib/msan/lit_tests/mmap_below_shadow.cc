// Test mmap behavior when map address is below shadow range.
// With MAP_FIXED, we crash.
// Without MAP_FIXED, we ignore the address hint and map somewhere in
// application range.

// RUN: %clangxx_msan -m64 -O0 -DFIXED=0 %s -o %t && %t
// RUN: %clangxx_msan -m64 -O0 -DFIXED=1 %s -o %t && not %t

#include <assert.h>
#include <stdint.h>
#include <sys/mman.h>

int main(void) {
  // Hint address just below shadow.
  uintptr_t hint = 0x1f0000000000ULL;
  const uintptr_t app_start = 0x600000000000ULL;
  uintptr_t p = (uintptr_t)mmap(
      (void *)hint, 4096, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS | (FIXED ? MAP_FIXED : 0), 0, 0);
  assert(p >= app_start);
  return 0;
}
