// Test that mmap (without MAP_FIXED) always returns valid application addresses.
// RUN: %clangxx_msan -O0 %s -o %t && %run %t
// RUN: %clangxx_msan -O0 -fsanitize-memory-track-origins %s -o %t && %run %t

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>
#include <stdio.h>

bool AddrIsApp(void *p) {
  uintptr_t addr = (uintptr_t)p;
#if defined(__FreeBSD__) && defined(__x86_64__)
  return addr < 0x010000000000ULL || addr >= 0x600000000000ULL;
#elif defined(__x86_64__)
  return addr >= 0x600000000000ULL;
#elif defined(__mips64)
  return addr >= 0x00e000000000ULL;
#endif
}

int main() {
  // Large enough to quickly exhaust the entire address space.
#if defined(__mips64)
  const size_t kMapSize = 0x100000000ULL;
#else
  const size_t kMapSize = 0x1000000000ULL;
#endif
  int success_count = 0;
  while (true) {
    void *p = mmap(0, kMapSize, PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    printf("%p\n", p);
    if (p == MAP_FAILED) {
      assert(errno == ENOMEM);
      break;
    }
    assert(AddrIsApp(p));
    success_count++;
  }
  printf("successful mappings: %d\n", success_count);
  assert(success_count > 5);
}
