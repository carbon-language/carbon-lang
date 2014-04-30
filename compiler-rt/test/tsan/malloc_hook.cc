// RUN: %clangxx_tsan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

static int malloc_count;
static int free_count;

extern "C" {
void __tsan_malloc_hook(void *ptr, size_t size) {
  (void)ptr;
  (void)size;
  __sync_fetch_and_add(&malloc_count, 1);
}

void __tsan_free_hook(void *ptr) {
  (void)ptr;
  __sync_fetch_and_add(&free_count, 1);
}
}

void *Thread1(void *x) {
  ((int*)x)[0]++;
  return 0;
}

void *Thread2(void *x) {
  sleep(1);
  ((int*)x)[0]++;
  return 0;
}

int main() {
  int *x = new int;
  pthread_t t[2];
  pthread_create(&t[0], 0, Thread1, x);
  pthread_create(&t[1], 0, Thread2, x);
  pthread_join(t[0], 0);
  pthread_join(t[1], 0);
  delete x;
  if (malloc_count == 0 || free_count == 0) {
    fprintf(stderr, "FAILED %d %d\n", malloc_count, free_count);
    exit(1);
  }
  fprintf(stderr, "DONE\n");
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NOT: FAILED
// CHECK: DONE
