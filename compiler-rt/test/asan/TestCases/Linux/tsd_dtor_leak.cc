// Regression test for a leak in tsd:
// https://code.google.com/p/address-sanitizer/issues/detail?id=233
// RUN: %clangxx_asan -O1 %s -pthread -o %t
// RUN: ASAN_OPTIONS=quarantine_size=1 %run %t
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sanitizer/allocator_interface.h>

static pthread_key_t tsd_key;

void *Thread(void *) {
  pthread_setspecific(tsd_key, malloc(10));
  return 0;
}

static volatile void *v;

void Dtor(void *tsd) {
  v = malloc(10000);
  free(tsd);
  free((void*)v);  // The bug was that this was leaking.
}

int main() {
  assert(0 == pthread_key_create(&tsd_key, Dtor));
  size_t old_heap_size = 0;
  for (int i = 0; i < 10; i++) {
    pthread_t t;
    pthread_create(&t, 0, Thread, 0);
    pthread_join(t, 0);
    size_t new_heap_size = __sanitizer_get_heap_size();
    fprintf(stderr, "heap size: new: %zd old: %zd\n", new_heap_size, old_heap_size);
    if (old_heap_size)
      assert(old_heap_size == new_heap_size);
    old_heap_size = new_heap_size;
  }
}
