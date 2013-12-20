// Test indirect call wrapping in MemorySanitizer runtime.

// RUN: %clangxx_msan -O0 -g -rdynamic %s -o %t && %t

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>

extern "C" void __msan_set_indirect_call_wrapper(uintptr_t);

bool done;

void *ThreadFn(void *) {
  printf("bad threadfn\n");
  return 0;
}

void *ThreadFn2(void *) {
  printf("good threadfn\n");
  done = true;
  return 0;
}

// ThreadFn is called indirectly from a wrapper function in MSan rtl and
// is subject to indirect call wrapping (it could be an native-to-translated
// edge).
extern "C" uintptr_t my_wrapper(uintptr_t f) {
  if (f == (uintptr_t)ThreadFn)
    return (uintptr_t)&ThreadFn2;
  return f;
}

int main(void) {
  __msan_set_indirect_call_wrapper((uintptr_t)my_wrapper);
  pthread_t t;
  pthread_create(&t, 0, ThreadFn, 0);
  pthread_join(t, 0);
  assert(done);
  return 0;
}
