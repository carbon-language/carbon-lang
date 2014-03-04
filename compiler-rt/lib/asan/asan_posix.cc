//===-- asan_posix.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Posix-specific details.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_POSIX

#include "asan_internal.h"
#include "asan_interceptors.h"
#include "asan_mapping.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_procmaps.h"

#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

namespace __asan {

void AsanOnSIGSEGV(int, void *siginfo, void *context) {
  uptr addr = (uptr)((siginfo_t*)siginfo)->si_addr;
  int code = (int)((siginfo_t*)siginfo)->si_code;
  // Write the first message using the bullet-proof write.
  if (13 != internal_write(2, "ASAN:SIGSEGV\n", 13)) Die();
  uptr pc, sp, bp;
  GetPcSpBp(context, &pc, &sp, &bp);

  // Access at a reasonable offset above SP, or slightly below it (to account
  // for x86_64 redzone, ARM push of multiple registers, etc) is probably a
  // stack overflow.
  // We also check si_code to filter out SEGV caused by something else other
  // then hitting the guard page or unmapped memory, like, for example,
  // unaligned memory access.
  if (addr + 128 > sp && addr < sp + 0xFFFF &&
      (code == si_SEGV_MAPERR || code == si_SEGV_ACCERR))
    ReportStackOverflow(pc, sp, bp, context, addr);
  else
    ReportSIGSEGV(pc, sp, bp, context, addr);
}

// ---------------------- TSD ---------------- {{{1

static pthread_key_t tsd_key;
static bool tsd_key_inited = false;
void AsanTSDInit(void (*destructor)(void *tsd)) {
  CHECK(!tsd_key_inited);
  tsd_key_inited = true;
  CHECK_EQ(0, pthread_key_create(&tsd_key, destructor));
}

void *AsanTSDGet() {
  CHECK(tsd_key_inited);
  return pthread_getspecific(tsd_key);
}

void AsanTSDSet(void *tsd) {
  CHECK(tsd_key_inited);
  pthread_setspecific(tsd_key, tsd);
}

void PlatformTSDDtor(void *tsd) {
  AsanThreadContext *context = (AsanThreadContext*)tsd;
  if (context->destructor_iterations > 1) {
    context->destructor_iterations--;
    CHECK_EQ(0, pthread_setspecific(tsd_key, tsd));
    return;
  }
  AsanThread::TSDDtor(tsd);
}
}  // namespace __asan

#endif  // SANITIZER_POSIX
