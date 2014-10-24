// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>

// Test for issue:
// https://code.google.com/p/thread-sanitizer/issues/detail?id=5

void *Thread(void *ptr) {
  *(int*)ptr = 42;
  return 0;
}

int main() {
  void *ptr = mmap(0, 128 << 10, PROT_READ|PROT_WRITE,
      MAP_32BIT|MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  fprintf(stderr, "ptr=%p\n", ptr);
  if (ptr == MAP_FAILED) {
    fprintf(stderr, "mmap failed: %d\n", errno);
    return 1;
  }
  if ((uintptr_t)ptr >= (1ull << 32)) {
    fprintf(stderr, "ptr is too high\n");
    return 1;
  }
  pthread_t t;
  pthread_create(&t, 0, Thread, ptr);
  sleep(1);
  *(int*)ptr = 42;
  pthread_join(t, 0);
  munmap(ptr, 128 << 10);
  fprintf(stderr, "DONE\n");
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: DONE

