// Test indirect call wrapping in MemorySanitizer runtime.

// RUN: %clangxx_msan -O0 -g -rdynamic %s -o %t && %run %t
// RUN: %clangxx_msan -O2 -g -rdynamic %s -o %t && %run %t

#include <assert.h>
#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

extern "C" void __msan_set_indirect_call_wrapper(uintptr_t);

bool pthread_create_done;

void *ThreadFn(void *) {
  printf("bad threadfn\n");
  return 0;
}

void *ThreadFn2(void *) {
  printf("good threadfn\n");
  pthread_create_done = true;
  return 0;
}

int my_gettimeofday(struct timeval *p, void *q) {
  p->tv_sec = 1;
  p->tv_usec = 2;
  return 42;
}

double my_lgamma(double x) {
  return x;
}

uintptr_t real_gettimeofday;
uintptr_t real_lgamma;

extern "C" uintptr_t my_wrapper(uintptr_t f) {
  if (f == (uintptr_t)ThreadFn) return (uintptr_t)&ThreadFn2;
  if (f == real_gettimeofday) return (uintptr_t)my_gettimeofday;
  if (f == real_lgamma) return (uintptr_t)my_lgamma;
  return f;
}

int main(void) {
  real_gettimeofday = (uintptr_t)dlsym(RTLD_NEXT, "gettimeofday");
  real_lgamma = (uintptr_t)dlsym(RTLD_NEXT, "lgamma");

  __msan_set_indirect_call_wrapper((uintptr_t)my_wrapper);

  // ThreadFn is called indirectly from a wrapper function in MSan rtl and
  // is subject to indirect call wrapping (it could be an native-to-translated
  // edge).
  pthread_t t;
  pthread_create(&t, 0, ThreadFn, 0);
  pthread_join(t, 0);
  assert(pthread_create_done);

  // gettimeofday is intercepted in msan_interceptors.cc and the real one (from
  // libc) is called indirectly.
  struct timeval tv;
  int res = gettimeofday(&tv, NULL);
  assert(tv.tv_sec == 1);
  assert(tv.tv_usec == 2);
  assert(res == 42);

  // lgamma is intercepted in sanitizer_common_interceptors.inc and is also
  // called indirectly.
  double dres = lgamma(1.1);
  assert(dres == 1.1);
  
  return 0;
}
