//===-- dd_interceptors.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "dd_rtl.h"
#include "interception/interception.h"
#include <pthread.h>

using namespace __dsan;

extern "C" void *__libc_malloc(uptr size);
extern "C" void __libc_free(void *ptr);

static __thread Thread *thr;

static void InitThread() {
  if (thr != 0)
    return;
  thr = (Thread*)InternalAlloc(sizeof(*thr));
  internal_memset(thr, 0, sizeof(*thr));
  ThreadInit(thr);
}

INTERCEPTOR(int, pthread_mutex_destroy, pthread_mutex_t *m) {
  InitThread();
  int res = REAL(pthread_mutex_destroy)(m);
  MutexDestroy(thr, (uptr)m);
  return res;
}

INTERCEPTOR(int, pthread_mutex_lock, pthread_mutex_t *m) {
  InitThread();
  int res = REAL(pthread_mutex_lock)(m);
  if (res == 0)
    MutexLock(thr, (uptr)m, true, false);
  return res;
}

INTERCEPTOR(int, pthread_mutex_trylock, pthread_mutex_t *m) {
  InitThread();
  int res = REAL(pthread_mutex_trylock)(m);
  if (res == 0)
    MutexLock(thr, (uptr)m, true, true);
  return res;
}

INTERCEPTOR(int, pthread_mutex_unlock, pthread_mutex_t *m) {
  InitThread();
  MutexUnlock(thr, (uptr)m, true);
  int res = REAL(pthread_mutex_unlock)(m);
  return res;
}

namespace __dsan {

void InitializeInterceptors() {
  INTERCEPT_FUNCTION(pthread_mutex_destroy);
  INTERCEPT_FUNCTION(pthread_mutex_lock);
  INTERCEPT_FUNCTION(pthread_mutex_trylock);
  INTERCEPT_FUNCTION(pthread_mutex_unlock);
}

}  // namespace __dsan

#if DYNAMIC
static void __local_dsan_init() __attribute__((constructor));
void __local_dsan_init() {
  __dsan::Initialize();
}
#else
__attribute__((section(".preinit_array"), used))
void (*__local_dsan_preinit)(void) = __dsan::Initialize;
#endif
