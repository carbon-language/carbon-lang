//===-- msan_linux.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// Linux-specific code.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_LINUX

#include "msan.h"
#include "msan_thread.h"

#include <elf.h>
#include <link.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <unwind.h>
#include <execinfo.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_procmaps.h"

namespace __msan {

static const uptr kMemBeg     = 0x600000000000;
static const uptr kMemEnd     = 0x7fffffffffff;
static const uptr kShadowBeg  = MEM_TO_SHADOW(kMemBeg);
static const uptr kShadowEnd  = MEM_TO_SHADOW(kMemEnd);
static const uptr kBad1Beg    = 0;
static const uptr kBad1End    = kShadowBeg - 1;
static const uptr kBad2Beg    = kShadowEnd + 1;
static const uptr kBad2End    = kMemBeg - 1;
static const uptr kOriginsBeg = kBad2Beg;
static const uptr kOriginsEnd = kBad2End;

bool InitShadow(bool prot1, bool prot2, bool map_shadow, bool init_origins) {
  if ((uptr) & InitShadow < kMemBeg) {
    Printf("FATAL: Code below application range: %p < %p. Non-PIE build?\n",
           &InitShadow, (void *)kMemBeg);
    return false;
  }

  VPrintf(1, "__msan_init %p\n", &__msan_init);
  VPrintf(1, "Memory   : %p %p\n", kMemBeg, kMemEnd);
  VPrintf(1, "Bad2     : %p %p\n", kBad2Beg, kBad2End);
  VPrintf(1, "Origins  : %p %p\n", kOriginsBeg, kOriginsEnd);
  VPrintf(1, "Shadow   : %p %p\n", kShadowBeg, kShadowEnd);
  VPrintf(1, "Bad1     : %p %p\n", kBad1Beg, kBad1End);

  if (!MemoryRangeIsAvailable(kShadowBeg,
                              init_origins ? kOriginsEnd : kShadowEnd) ||
      (prot1 && !MemoryRangeIsAvailable(kBad1Beg, kBad1End)) ||
      (prot2 && !MemoryRangeIsAvailable(kBad2Beg, kBad2End))) {
    Printf("FATAL: Shadow memory range is not available.\n");
    return false;
  }

  if (prot1 && !Mprotect(kBad1Beg, kBad1End - kBad1Beg))
    return false;
  if (prot2 && !Mprotect(kBad2Beg, kBad2End - kBad2Beg))
    return false;
  if (map_shadow) {
    void *shadow = MmapFixedNoReserve(kShadowBeg, kShadowEnd - kShadowBeg);
    if (shadow != (void*)kShadowBeg) return false;
  }
  if (init_origins) {
    void *origins = MmapFixedNoReserve(kOriginsBeg, kOriginsEnd - kOriginsBeg);
    if (origins != (void*)kOriginsBeg) return false;
  }
  return true;
}

void MsanDie() {
  if (death_callback)
    death_callback();
  _exit(flags()->exit_code);
}

static void MsanAtExit(void) {
  if (msan_report_count > 0) {
    ReportAtExitStatistics();
    if (flags()->exit_code)
      _exit(flags()->exit_code);
  }
}

void InstallAtExitHandler() {
  atexit(MsanAtExit);
}

// ---------------------- TSD ---------------- {{{1

static pthread_key_t tsd_key;
static bool tsd_key_inited = false;
void MsanTSDInit(void (*destructor)(void *tsd)) {
  CHECK(!tsd_key_inited);
  tsd_key_inited = true;
  CHECK_EQ(0, pthread_key_create(&tsd_key, destructor));
}

void *MsanTSDGet() {
  CHECK(tsd_key_inited);
  return pthread_getspecific(tsd_key);
}

void MsanTSDSet(void *tsd) {
  CHECK(tsd_key_inited);
  pthread_setspecific(tsd_key, tsd);
}

void MsanTSDDtor(void *tsd) {
  MsanThread *t = (MsanThread*)tsd;
  if (t->destructor_iterations_ > 1) {
    t->destructor_iterations_--;
    CHECK_EQ(0, pthread_setspecific(tsd_key, tsd));
    return;
  }
  MsanThread::TSDDtor(tsd);
}

}  // namespace __msan

#endif  // __linux__
