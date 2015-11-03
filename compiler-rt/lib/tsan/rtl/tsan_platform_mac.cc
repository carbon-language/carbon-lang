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
void InitializeShadowMemory() {
  uptr shadow = (uptr)MmapFixedNoReserve(kShadowBeg,
    kShadowEnd - kShadowBeg);
  if (shadow != kShadowBeg) {
    Printf("FATAL: ThreadSanitizer can not mmap the shadow memory\n");
    Printf("FATAL: Make sure to compile with -fPIE and "
           "to link with -pie.\n");
    Die();
  }
  if (common_flags()->use_madv_dontdump)
    DontDumpShadowMemory(kShadowBeg, kShadowEnd - kShadowBeg);
  DPrintf("kShadow %zx-%zx (%zuGB)\n",
      kShadowBeg, kShadowEnd,
      (kShadowEnd - kShadowBeg) >> 30);
  DPrintf("kAppMem %zx-%zx (%zuGB)\n",
      kAppMemBeg, kAppMemEnd,
      (kAppMemEnd - kAppMemBeg) >> 30);
}
#endif

void InitializePlatform() {
  DisableCoreDumperIfNecessary();
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
