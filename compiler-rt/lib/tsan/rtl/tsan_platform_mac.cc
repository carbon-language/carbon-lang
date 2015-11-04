//===-- tsan_platform_mac.cc ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Mac-specific code.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_MAC

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "tsan_flags.h"

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>

namespace __tsan {

uptr GetShadowMemoryConsumption() {
  return 0;
}

void FlushShadowMemory() {
}

void WriteMemoryProfile(char *buf, uptr buf_size, uptr nthread, uptr nlive) {
}

#ifndef SANITIZER_GO
void InitializeShadowMemoryPlatform() { }
#endif

// On OS X, GCD worker threads are created without a call to pthread_create. We
// need to properly register these threads with ThreadCreate and ThreadStart.
// These threads don't have a parent thread, as they are created "spuriously".
// We're using a libpthread API that notifies us about a newly created thread.
// The `thread == pthread_self()` check indicates this is actually a worker
// thread. If it's just a regular thread, this hook is called on the parent
// thread.
typedef void (*pthread_introspection_hook_t)(unsigned int event,
                                             pthread_t thread, void *addr,
                                             size_t size);
extern "C" pthread_introspection_hook_t pthread_introspection_hook_install(
    pthread_introspection_hook_t hook);
static const uptr PTHREAD_INTROSPECTION_THREAD_CREATE = 1;
static const uptr PTHREAD_INTROSPECTION_THREAD_DESTROY = 4;
static pthread_introspection_hook_t prev_pthread_introspection_hook;
static void my_pthread_introspection_hook(unsigned int event, pthread_t thread,
                                          void *addr, size_t size) {
  if (event == PTHREAD_INTROSPECTION_THREAD_CREATE) {
    if (thread == pthread_self()) {
      // The current thread is a newly created GCD worker thread.
      ThreadState *parent_thread_state = nullptr;  // No parent.
      int tid = ThreadCreate(parent_thread_state, 0, (uptr)thread, true);
      CHECK_NE(tid, 0);
      ThreadState *thr = cur_thread();
      ThreadStart(thr, tid, GetTid());
    }
  } else if (event == PTHREAD_INTROSPECTION_THREAD_DESTROY) {
    ThreadState *thr = cur_thread();
    if (thr->tctx->parent_tid == kInvalidTid) {
      DestroyThreadState();
    }
  }

  if (prev_pthread_introspection_hook != nullptr)
    prev_pthread_introspection_hook(event, thread, addr, size);
}

void InitializePlatform() {
  DisableCoreDumperIfNecessary();
#ifndef SANITIZER_GO
  CheckAndProtect();
#endif

  prev_pthread_introspection_hook =
      pthread_introspection_hook_install(&my_pthread_introspection_hook);
}

#ifndef SANITIZER_GO
// Note: this function runs with async signals enabled,
// so it must not touch any tsan state.
int call_pthread_cancel_with_cleanup(int(*fn)(void *c, void *m,
    void *abstime), void *c, void *m, void *abstime,
    void(*cleanup)(void *arg), void *arg) {
  // pthread_cleanup_push/pop are hardcore macros mess.
  // We can't intercept nor call them w/o including pthread.h.
  int res;
  pthread_cleanup_push(cleanup, arg);
  res = fn(c, m, abstime);
  pthread_cleanup_pop(0);
  return res;
}
#endif

bool IsGlobalVar(uptr addr) {
  return false;
}

}  // namespace __tsan

#endif  // SANITIZER_MAC
