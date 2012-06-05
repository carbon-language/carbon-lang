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
#include "asan_procmaps.h"
#include "asan_stack.h"
#include "asan_thread_registry.h"
#include "sanitizer_common/sanitizer_libc.h"

#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#ifdef ANDROID
#include <sys/atomics.h>
#endif

// Should not add dependency on libstdc++,
// since most of the stuff here is inlinable.
#include <algorithm>

using namespace __sanitizer;

static const uptr kAltStackSize = SIGSTKSZ * 4;  // SIGSTKSZ is not enough.

namespace __asan {

static inline bool IntervalsAreSeparate(uptr start1, uptr end1,
                                        uptr start2, uptr end2) {
  CHECK(start1 <= end1);
  CHECK(start2 <= end2);
  return (end1 < start2) || (end2 < start1);
}

// FIXME: this is thread-unsafe, but should not cause problems most of the time.
// When the shadow is mapped only a single thread usually exists (plus maybe
// several worker threads on Mac, which aren't expected to map big chunks of
// memory).
bool AsanShadowRangeIsAvailable() {
  AsanProcMaps procmaps;
  uptr start, end;
  uptr shadow_start = kLowShadowBeg;
  if (kLowShadowBeg > 0) shadow_start -= kMmapGranularity;
  uptr shadow_end = kHighShadowEnd;
  while (procmaps.Next(&start, &end,
                       /*offset*/0, /*filename*/0, /*filename_size*/0)) {
    if (!IntervalsAreSeparate(start, end, shadow_start, shadow_end))
      return false;
  }
  return true;
}

static void MaybeInstallSigaction(int signum,
                                  void (*handler)(int, siginfo_t *, void *)) {
  if (!AsanInterceptsSignal(signum))
    return;
  struct sigaction sigact;
  REAL(memset)(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = handler;
  sigact.sa_flags = SA_SIGINFO;
  if (FLAG_use_sigaltstack) sigact.sa_flags |= SA_ONSTACK;
  CHECK(0 == REAL(sigaction)(signum, &sigact, 0));
  if (FLAG_v >= 1) {
    Report("Installed the sigaction for signal %d\n", signum);
  }
}

static void     ASAN_OnSIGSEGV(int, siginfo_t *siginfo, void *context) {
  uptr addr = (uptr)siginfo->si_addr;
  // Write the first message using the bullet-proof write.
  if (13 != internal_write(2, "ASAN:SIGSEGV\n", 13)) AsanDie();
  uptr pc, sp, bp;
  GetPcSpBp(context, &pc, &sp, &bp);
  Report("ERROR: AddressSanitizer crashed on unknown address %p"
         " (pc %p sp %p bp %p T%d)\n",
         addr, pc, sp, bp,
         asanThreadRegistry().GetCurrentTidOrMinusOne());
  Printf("AddressSanitizer can not provide additional info. ABORTING\n");
  GET_STACK_TRACE_WITH_PC_AND_BP(kStackTraceMax, pc, bp);
  stack.PrintStack();
  ShowStatsAndAbort();
}

void SetAlternateSignalStack() {
  stack_t altstack, oldstack;
  CHECK(0 == sigaltstack(0, &oldstack));
  // If the alternate stack is already in place, do nothing.
  if ((oldstack.ss_flags & SS_DISABLE) == 0) return;
  // TODO(glider): the mapped stack should have the MAP_STACK flag in the
  // future. It is not required by man 2 sigaltstack now (they're using
  // malloc()).
  void* base = AsanMmapSomewhereOrDie(kAltStackSize, __FUNCTION__);
  altstack.ss_sp = base;
  altstack.ss_flags = 0;
  altstack.ss_size = kAltStackSize;
  CHECK(0 == sigaltstack(&altstack, 0));
  if (FLAG_v > 0) {
    Report("Alternative stack for T%d set: [%p,%p)\n",
           asanThreadRegistry().GetCurrentTidOrMinusOne(),
           altstack.ss_sp, (char*)altstack.ss_sp + altstack.ss_size);
  }
}

void UnsetAlternateSignalStack() {
  stack_t altstack, oldstack;
  altstack.ss_sp = 0;
  altstack.ss_flags = SS_DISABLE;
  altstack.ss_size = 0;
  CHECK(0 == sigaltstack(&altstack, &oldstack));
  AsanUnmapOrDie(oldstack.ss_sp, oldstack.ss_size);
}

void InstallSignalHandlers() {
  // Set the alternate signal stack for the main thread.
  // This will cause SetAlternateSignalStack to be called twice, but the stack
  // will be actually set only once.
  if (FLAG_use_sigaltstack) SetAlternateSignalStack();
  MaybeInstallSigaction(SIGSEGV, ASAN_OnSIGSEGV);
  MaybeInstallSigaction(SIGBUS, ASAN_OnSIGSEGV);
}

void AsanDisableCoreDumper() {
  struct rlimit nocore;
  nocore.rlim_cur = 0;
  nocore.rlim_max = 0;
  setrlimit(RLIMIT_CORE, &nocore);
}

void AsanDumpProcessMap() {
  AsanProcMaps proc_maps;
  uptr start, end;
  const sptr kBufSize = 4095;
  char filename[kBufSize];
  Report("Process memory map follows:\n");
  while (proc_maps.Next(&start, &end, /* file_offset */0,
                        filename, kBufSize)) {
    Printf("\t%p-%p\t%s\n", (void*)start, (void*)end, filename);
  }
  Report("End of process memory map.\n");
}

int GetPid() {
  return getpid();
}

uptr GetThreadSelf() {
  return (uptr)pthread_self();
}

void SleepForSeconds(int seconds) {
  sleep(seconds);
}

void Exit(int exitcode) {
  _exit(exitcode);
}

void Abort() {
  abort();
}

int Atexit(void (*function)(void)) {
  return atexit(function);
}

int AtomicInc(int *a) {
#ifdef ANDROID
  return __atomic_inc(a) + 1;
#else
  return __sync_add_and_fetch(a, 1);
#endif
}

u16 AtomicExchange(u16 *a, u16 new_val) {
  return __sync_lock_test_and_set(a, new_val);
}

void SortArray(uptr *array, uptr size) {
  std::sort(array, array + size);
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
