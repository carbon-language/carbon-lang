// Test that mmap (without MAP_FIXED) always returns valid application addresses.
// RUN: %clangxx_msan -O0 %s -o %t && %run %t
// RUN: %clangxx_msan -O0 -fsanitize-memory-track-origins %s -o %t && %run %t

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include "test.h"

bool AddrIsApp(void *p) {
  uintptr_t addr = (uintptr_t)p;
#if defined(__FreeBSD__) && defined(__x86_64__)
  return addr < 0x010000000000ULL || addr >= 0x600000000000ULL;
#elif defined(__x86_64__)
  return addr >= 0x600000000000ULL;
#elif defined(__mips64)
  return addr >= 0x00e000000000ULL;
#elif defined(__powerpc64__)
  return addr < 0x000100000000ULL || addr >= 0x300000000000ULL;
#elif defined(__aarch64__)
  unsigned long vma = SystemVMA();
  if (vma == 39)
    return (addr >= 0x5500000000ULL && addr < 0x5600000000ULL) ||
           (addr > 0x7000000000ULL);
  else if (vma == 42)
    return (addr >= 0x2aa00000000ULL && addr < 0x2ab00000000ULL) ||
           (addr > 0x3f000000000ULL);
  else {
    fprintf(stderr, "unsupported vma: %lu\n", vma);
    exit(1);
  }
#endif
}

int main() {
  // Large enough to quickly exhaust the entire address space.
#if defined(__mips64) || defined(__aarch64__)
  const size_t kMapSize = 0x100000000ULL;
#else
  const size_t kMapSize = 0x1000000000ULL;
#endif
  int success_count = 0;
  while (true) {
    void *p = mmap(0, kMapSize, PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
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
