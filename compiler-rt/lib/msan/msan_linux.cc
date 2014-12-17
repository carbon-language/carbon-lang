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
// Linux- and FreeBSD-specific code.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_FREEBSD || SANITIZER_LINUX

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

void ReportMapRange(const char *descr, uptr beg, uptr size) {
  if (size > 0) {
    uptr end = beg + size - 1;
    VPrintf(1, "%s : %p - %p\n", descr, beg, end);
  }
}

static bool CheckMemoryRangeAvailability(uptr beg, uptr size) {
  if (size > 0) {
    uptr end = beg + size - 1;
    if (!MemoryRangeIsAvailable(beg, end)) {
      Printf("FATAL: Memory range %p - %p is not available.\n", beg, end);
      return false;
    }
  }
  return true;
}

static bool ProtectMemoryRange(uptr beg, uptr size) {
  if (size > 0) {
    uptr end = beg + size - 1;
    if (!Mprotect(beg, size)) {
      Printf("FATAL: Cannot protect memory range %p - %p.\n", beg, end);
      return false;
    }
  }
  return true;
}

bool InitShadow(bool map_shadow, bool init_origins) {
  // Let user know mapping parameters first.
  VPrintf(1, "__msan_init %p\n", &__msan_init);
  ReportMapRange("Low Memory ", kLowMemBeg, kLowMemSize);
  ReportMapRange("Bad1       ", kBad1Beg, kBad1Size);
  ReportMapRange("Shadow     ", kShadowBeg, kShadowSize);
  ReportMapRange("Bad2       ", kBad2Beg, kBad2Size);
  ReportMapRange("Origins    ", kOriginsBeg, kOriginsSize);
  ReportMapRange("Bad3       ", kBad3Beg, kBad3Size);
  ReportMapRange("High Memory", kHighMemBeg, kHighMemSize);

  // Check mapping sanity (the invariant).
  CHECK_EQ(kLowMemBeg, 0);
  CHECK_EQ(kBad1Beg, kLowMemBeg + kLowMemSize);
  CHECK_EQ(kShadowBeg, kBad1Beg + kBad1Size);
  CHECK_GT(kShadowSize, 0);
  CHECK_GE(kShadowSize, kLowMemSize + kHighMemSize);
  CHECK_EQ(kBad2Beg, kShadowBeg + kShadowSize);
  CHECK_EQ(kOriginsBeg, kBad2Beg + kBad2Size);
  CHECK_EQ(kOriginsSize, kShadowSize);
  CHECK_EQ(kBad3Beg, kOriginsBeg + kOriginsSize);
  CHECK_EQ(kHighMemBeg, kBad3Beg + kBad3Size);
  CHECK_GT(kHighMemSize, 0);
  CHECK_GE(kHighMemBeg + kHighMemSize, kHighMemBeg);  // Tests for no overflow.

  if (kLowMemSize > 0) {
    CHECK(MEM_IS_SHADOW(MEM_TO_SHADOW(kLowMemBeg)));
    CHECK(MEM_IS_SHADOW(MEM_TO_SHADOW(kLowMemBeg + kLowMemSize - 1)));
    CHECK(MEM_IS_ORIGIN(MEM_TO_ORIGIN(kLowMemBeg)));
    CHECK(MEM_IS_ORIGIN(MEM_TO_ORIGIN(kLowMemBeg + kLowMemSize - 1)));
  }
  CHECK(MEM_IS_SHADOW(MEM_TO_SHADOW(kHighMemBeg)));
  CHECK(MEM_IS_SHADOW(MEM_TO_SHADOW(kHighMemBeg + kHighMemSize - 1)));
  CHECK(MEM_IS_ORIGIN(MEM_TO_ORIGIN(kHighMemBeg)));
  CHECK(MEM_IS_ORIGIN(MEM_TO_ORIGIN(kHighMemBeg + kHighMemSize - 1)));

  if (!MEM_IS_APP(&__msan_init)) {
    Printf("FATAL: Code %p is out of application range. Non-PIE build?\n",
           (uptr)&__msan_init);
    return false;
  }

  if (!CheckMemoryRangeAvailability(kShadowBeg, kShadowSize) ||
      (init_origins &&
        !CheckMemoryRangeAvailability(kOriginsBeg, kOriginsSize)) ||
      !CheckMemoryRangeAvailability(kBad1Beg, kBad1Size) ||
      !CheckMemoryRangeAvailability(kBad2Beg, kBad2Size) ||
      !CheckMemoryRangeAvailability(kBad3Beg, kBad3Size)) {
    return false;
  }

  if (!ProtectMemoryRange(kBad1Beg, kBad1Size) ||
      !ProtectMemoryRange(kBad2Beg, kBad2Size) ||
      !ProtectMemoryRange(kBad3Beg, kBad3Size)) {
    return false;
  }

  if (map_shadow) {
    void *shadow = MmapFixedNoReserve(kShadowBeg, kShadowSize);
    if (shadow != (void*)kShadowBeg) return false;
  }
  if (init_origins) {
    void *origins = MmapFixedNoReserve(kOriginsBeg, kOriginsSize);
    if (origins != (void*)kOriginsBeg) return false;
  }
  return true;
}

void MsanDie() {
  if (common_flags()->coverage)
    __sanitizer_cov_dump();
  if (death_callback)
    death_callback();
  _exit(flags()->exit_code);
}

static void MsanAtExit(void) {
  if (flags()->print_stats && (flags()->atexit || msan_report_count > 0))
    ReportStats();
  if (msan_report_count > 0) {
    ReportAtExitStatistics();
    if (flags()->exit_code) _exit(flags()->exit_code);
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

static THREADLOCAL MsanThread* msan_current_thread;

MsanThread *GetCurrentThread() {
  return msan_current_thread;
}

void SetCurrentThread(MsanThread *t) {
  // Make sure we do not reset the current MsanThread.
  CHECK_EQ(0, msan_current_thread);
  msan_current_thread = t;
  // Make sure that MsanTSDDtor gets called at the end.
  CHECK(tsd_key_inited);
  pthread_setspecific(tsd_key, (void *)t);
}

void MsanTSDDtor(void *tsd) {
  MsanThread *t = (MsanThread*)tsd;
  if (t->destructor_iterations_ > 1) {
    t->destructor_iterations_--;
    CHECK_EQ(0, pthread_setspecific(tsd_key, tsd));
    return;
  }
  msan_current_thread = nullptr;
  // Make sure that signal handler can not see a stale current thread pointer.
  atomic_signal_fence(memory_order_seq_cst);
  MsanThread::TSDDtor(tsd);
}

}  // namespace __msan

#endif  // SANITIZER_FREEBSD || SANITIZER_LINUX
