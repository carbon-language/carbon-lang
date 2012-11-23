//===-- asan_linux.cc -----------------------------------------------------===//
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
#if defined(__linux__) || defined(__APPLE__)

#include "asan_internal.h"
#include "asan_interceptors.h"
#include "asan_mapping.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_thread_registry.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_procmaps.h"

#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <ucontext.h>
#include <unistd.h>

static const uptr kAltStackSize = SIGSTKSZ * 4;  // SIGSTKSZ is not enough.

namespace __asan {

static void MaybeInstallSigaction(int signum,
                                  void (*handler)(int, siginfo_t *, void *)) {
  if (!AsanInterceptsSignal(signum))
    return;
  struct sigaction sigact;
  REAL(memset)(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = handler;
  sigact.sa_flags = SA_SIGINFO;
  if (flags()->use_sigaltstack) sigact.sa_flags |= SA_ONSTACK;
  CHECK(0 == REAL(sigaction)(signum, &sigact, 0));
  if (flags()->verbosity >= 1) {
    Report("Installed the sigaction for signal %d\n", signum);
  }
}

static void     ASAN_OnSIGSEGV(int, siginfo_t *siginfo, void *context) {
  uptr addr = (uptr)siginfo->si_addr;
  // Write the first message using the bullet-proof write.
  if (13 != internal_write(2, "ASAN:SIGSEGV\n", 13)) Die();
  uptr pc, sp, bp;
  GetPcSpBp(context, &pc, &sp, &bp);
  ReportSIGSEGV(pc, sp, bp, addr);
}

void SetAlternateSignalStack() {
  stack_t altstack, oldstack;
  CHECK(0 == sigaltstack(0, &oldstack));
  // If the alternate stack is already in place, do nothing.
  if ((oldstack.ss_flags & SS_DISABLE) == 0) return;
  // TODO(glider): the mapped stack should have the MAP_STACK flag in the
  // future. It is not required by man 2 sigaltstack now (they're using
  // malloc()).
  void* base = MmapOrDie(kAltStackSize, __FUNCTION__);
  altstack.ss_sp = base;
  altstack.ss_flags = 0;
  altstack.ss_size = kAltStackSize;
  CHECK(0 == sigaltstack(&altstack, 0));
  if (flags()->verbosity > 0) {
    Report("Alternative stack for T%d set: [%p,%p)\n",
           asanThreadRegistry().GetCurrentTidOrInvalid(),
           altstack.ss_sp, (char*)altstack.ss_sp + altstack.ss_size);
  }
}

void UnsetAlternateSignalStack() {
  stack_t altstack, oldstack;
  altstack.ss_sp = 0;
  altstack.ss_flags = SS_DISABLE;
  altstack.ss_size = 0;
  CHECK(0 == sigaltstack(&altstack, &oldstack));
  UnmapOrDie(oldstack.ss_sp, oldstack.ss_size);
}

void InstallSignalHandlers() {
  // Set the alternate signal stack for the main thread.
  // This will cause SetAlternateSignalStack to be called twice, but the stack
  // will be actually set only once.
  if (flags()->use_sigaltstack) SetAlternateSignalStack();
  MaybeInstallSigaction(SIGSEGV, ASAN_OnSIGSEGV);
  MaybeInstallSigaction(SIGBUS, ASAN_OnSIGSEGV);
}

void ClearShadowMemoryForContext(void *context) {
  ucontext_t *ucp = (ucontext_t*)context;
  uptr sp = (uptr)ucp->uc_stack.ss_sp;
  uptr size = ucp->uc_stack.ss_size;
  // Align to page size.
  uptr bottom = sp & ~(kPageSize - 1);
  size += sp - bottom;
  size = RoundUpTo(size, kPageSize);
  PoisonShadow(bottom, size, 0);
}

// ---------------------- TSD ---------------- {{{1

static pthread_key_t tsd_key;
static bool tsd_key_inited = false;
void AsanTSDInit(void (*destructor)(void *tsd)) {
  CHECK(!tsd_key_inited);
  tsd_key_inited = true;
  CHECK(0 == pthread_key_create(&tsd_key, destructor));
}

void *AsanTSDGet() {
  CHECK(tsd_key_inited);
  return pthread_getspecific(tsd_key);
}

void AsanTSDSet(void *tsd) {
  CHECK(tsd_key_inited);
  pthread_setspecific(tsd_key, tsd);
}

}  // namespace __asan

#endif  // __linux__ || __APPLE_
