// Test that LargeAllocator unpoisons memory before releasing it to the OS.
// RUN: %clangxx_asan %s -o %t
// RUN: ASAN_OPTIONS=quarantine_size=1 %t

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

int main() {
  void *p = malloc(1024 * 1024);
  free(p);

  char *q = (char *)mmap(p, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
  assert(q);
  assert(q <= p);
  assert(q + 4096 > p);

  memset(q, 42, 4096);

  munmap(q, 4096);
  return 0;
}
