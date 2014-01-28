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
#if SANITIZER_LINUX || SANITIZER_MAC

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

static void MaybeInstallSigaction(int signum,
                                  void (*handler)(int, siginfo_t *, void *)) {
  if (!AsanInterceptsSignal(signum))
    return;
  struct sigaction sigact;
  REAL(memset)(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = handler;
  sigact.sa_flags = SA_SIGINFO;
  if (common_flags()->use_sigaltstack) sigact.sa_flags |= SA_ONSTACK;
  CHECK_EQ(0, REAL(sigaction)(signum, &sigact, 0));
  VReport(1, "Installed the sigaction for signal %d\n", signum);
}

static void     ASAN_OnSIGSEGV(int, siginfo_t *siginfo, void *context) {
  uptr addr = (uptr)siginfo->si_addr;
  // Write the first message using the bullet-proof write.
  if (13 != internal_write(2, "ASAN:SIGSEGV\n", 13)) Die();
  uptr pc, sp, bp;
  GetPcSpBp(context, &pc, &sp, &bp);
  ReportSIGSEGV(pc, sp, bp, addr);
}

void InstallSignalHandlers() {
  // Set the alternate signal stack for the main thread.
  // This will cause SetAlternateSignalStack to be called twice, but the stack
  // will be actually set only once.
  if (common_flags()->use_sigaltstack) SetAlternateSignalStack();
  MaybeInstallSigaction(SIGSEGV, ASAN_OnSIGSEGV);
  MaybeInstallSigaction(SIGBUS, ASAN_OnSIGSEGV);
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

#endif  // SANITIZER_LINUX || SANITIZER_MAC
