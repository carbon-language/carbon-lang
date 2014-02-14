// Test that LargeAllocator unpoisons memory before releasing it to the OS.
// RUN: %clangxx_asan %s -o %t
// The memory is released only when the deallocated chunk leaves the quarantine,
// otherwise the mmap(p, ...) call overwrites the malloc header.
// RUN: ASAN_OPTIONS=quarantine_size=1 %t

#include <assert.h>
#include <malloc.h>
#include <string.h>
#include <sys/mman.h>

int main() {
  const int kPageSize = 4096;
  void *p = memalign(kPageSize, 1024 * 1024);
  free(p);

  char *q = (char *)mmap(p, kPageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, 0, 0);
  assert(q == p);

  memset(q, 42, kPageSize);

  munmap(q, kPageSize);
  return 0;
}
