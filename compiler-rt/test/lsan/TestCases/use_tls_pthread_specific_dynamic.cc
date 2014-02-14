// Test that dynamically allocated thread-specific storage is included in the root set.
// RUN: LSAN_BASE="report_objects=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_tls=0" not %t 2>&1 | FileCheck %s
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_tls=1" %t 2>&1
// RUN: LSAN_OPTIONS="" %t 2>&1

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// From glibc: this many keys are stored in the thread descriptor directly.
const unsigned PTHREAD_KEY_2NDLEVEL_SIZE = 32;

int main() {
  static const unsigned kDummyKeysCount = PTHREAD_KEY_2NDLEVEL_SIZE;
  int res;
  pthread_key_t dummy_keys[kDummyKeysCount];
  for (unsigned i = 0; i < kDummyKeysCount; i++) {
    res = pthread_key_create(&dummy_keys[i], NULL);
    assert(res == 0);
  }
  pthread_key_t key;
  res = pthread_key_create(&key, NULL);
  assert(key >= PTHREAD_KEY_2NDLEVEL_SIZE);
  assert(res == 0);
  void *p  = malloc(1337);
  res = pthread_setspecific(key, p);
  assert(res == 0);
  fprintf(stderr, "Test alloc: %p.\n", p);
  return 0;
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
