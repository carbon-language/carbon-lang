// RUN: %clangxx_asan -O0 %s -DTHREAD -o %t && ASAN_OPTIONS=use_sigaltstack=1 not %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -DTHREAD -o %t && ASAN_OPTIONS=use_sigaltstack=1 not %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -DTHREAD -o %t && ASAN_OPTIONS=use_sigaltstack=1 not %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -DTHREAD -o %t && ASAN_OPTIONS=use_sigaltstack=1 not %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdlib.h>
#include <pthread.h>

const int BS = 1024;
volatile char x;

void large_frame_func(char *p, int level) {
  char buf[BS];
  if (p)
    assert(p - buf >= BS);
  buf[rand() % BS] = 1;
  buf[rand() % BS] = 2;
  x = buf[rand() % BS];
  volatile int y = 1;
  if (y)
    large_frame_func(buf, level + 1);
  // CHECK: {{stack-overflow on address 0x.* \(pc 0x.* sp 0x.* bp 0x.* T.*\)}}
  // Frame 0 may be anywhere (in rand(), for example).
  // CHECK: {{    #. 0x.* in large_frame_func.*stack-overflow.cc:}}[[@LINE-3]]
}

void *ThreadFn(void* unused) {
  large_frame_func(0, 0);
  return 0;
}

int main(int argc, char **argv) {
#ifdef THREAD
  pthread_t t;
  pthread_create(&t, 0, ThreadFn, 0);
  pthread_join(t, 0);
#else
  large_frame_func(0, 0);
#endif
  return 0;
}
