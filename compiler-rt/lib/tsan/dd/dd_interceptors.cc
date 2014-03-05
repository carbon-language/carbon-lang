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
  MutexDestroy(thr, (uptr)m);
  return REAL(pthread_mutex_destroy)(m);
}

INTERCEPTOR(int, pthread_mutex_lock, pthread_mutex_t *m) {
  InitThread();
  MutexLock(thr, (uptr)m, true, false);
  return REAL(pthread_mutex_lock)(m);
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
  return REAL(pthread_mutex_unlock)(m);
}

INTERCEPTOR(int, pthread_spin_destroy, pthread_spinlock_t *m) {
  InitThread();
  int res = REAL(pthread_spin_destroy)(m);
  MutexDestroy(thr, (uptr)m);
  return res;
}

INTERCEPTOR(int, pthread_spin_lock, pthread_spinlock_t *m) {
  InitThread();
  MutexLock(thr, (uptr)m, true, false);
  return REAL(pthread_spin_lock)(m);
}

INTERCEPTOR(int, pthread_spin_trylock, pthread_spinlock_t *m) {
  InitThread();
  int res = REAL(pthread_spin_trylock)(m);
  if (res == 0)
    MutexLock(thr, (uptr)m, true, true);
  return res;
}

INTERCEPTOR(int, pthread_spin_unlock, pthread_spinlock_t *m) {
  InitThread();
  MutexUnlock(thr, (uptr)m, true);
  return REAL(pthread_spin_unlock)(m);
}

INTERCEPTOR(int, pthread_rwlock_destroy, pthread_rwlock_t *m) {
  InitThread();
  MutexDestroy(thr, (uptr)m);
  return REAL(pthread_rwlock_destroy)(m);
}

INTERCEPTOR(int, pthread_rwlock_rdlock, pthread_rwlock_t *m) {
  InitThread();
  MutexLock(thr, (uptr)m, false, false);
  return REAL(pthread_rwlock_rdlock)(m);
}

INTERCEPTOR(int, pthread_rwlock_tryrdlock, pthread_rwlock_t *m) {
  InitThread();
  int res = REAL(pthread_rwlock_tryrdlock)(m);
  if (res == 0)
    MutexLock(thr, (uptr)m, false, true);
  return res;
}

INTERCEPTOR(int, pthread_rwlock_timedrdlock, pthread_rwlock_t *m,
    const timespec *abstime) {
  InitThread();
  int res = REAL(pthread_rwlock_timedrdlock)(m, abstime);
  if (res == 0)
    MutexLock(thr, (uptr)m, false, true);
  return res;
}

INTERCEPTOR(int, pthread_rwlock_wrlock, pthread_rwlock_t *m) {
  InitThread();
  MutexLock(thr, (uptr)m, true, false);
  return REAL(pthread_rwlock_wrlock)(m);
}

INTERCEPTOR(int, pthread_rwlock_trywrlock, pthread_rwlock_t *m) {
  InitThread();
  int res = REAL(pthread_rwlock_trywrlock)(m);
  if (res == 0)
    MutexLock(thr, (uptr)m, true, true);
  return res;
}

INTERCEPTOR(int, pthread_rwlock_timedwrlock, pthread_rwlock_t *m,
    const timespec *abstime) {
  InitThread();
  int res = REAL(pthread_rwlock_timedwrlock)(m, abstime);
  if (res == 0)
    MutexLock(thr, (uptr)m, true, true);
  return res;
}

INTERCEPTOR(int, pthread_rwlock_unlock, pthread_rwlock_t *m) {
  InitThread();
  MutexUnlock(thr, (uptr)m, true);  // note: not necessary write unlock
  return REAL(pthread_rwlock_unlock)(m);
}

namespace __dsan {

void InitializeInterceptors() {
  INTERCEPT_FUNCTION(pthread_mutex_destroy);
  INTERCEPT_FUNCTION(pthread_mutex_lock);
  INTERCEPT_FUNCTION(pthread_mutex_trylock);
  INTERCEPT_FUNCTION(pthread_mutex_unlock);

  INTERCEPT_FUNCTION(pthread_spin_destroy);
  INTERCEPT_FUNCTION(pthread_spin_lock);
  INTERCEPT_FUNCTION(pthread_spin_trylock);
  INTERCEPT_FUNCTION(pthread_spin_unlock);

  INTERCEPT_FUNCTION(pthread_rwlock_destroy);
  INTERCEPT_FUNCTION(pthread_rwlock_rdlock);
  INTERCEPT_FUNCTION(pthread_rwlock_tryrdlock);
  INTERCEPT_FUNCTION(pthread_rwlock_timedrdlock);
  INTERCEPT_FUNCTION(pthread_rwlock_wrlock);
  INTERCEPT_FUNCTION(pthread_rwlock_trywrlock);
  INTERCEPT_FUNCTION(pthread_rwlock_timedwrlock);
  INTERCEPT_FUNCTION(pthread_rwlock_unlock);
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
