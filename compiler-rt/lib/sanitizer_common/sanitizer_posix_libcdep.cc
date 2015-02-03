//===-- sanitizer_posix_libcdep.cc ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements libc-dependent POSIX-specific functions
// from sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_POSIX
#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_platform_limits_posix.h"
#include "sanitizer_stacktrace.h"

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

namespace __sanitizer {

u32 GetUid() {
  return getuid();
}

uptr GetThreadSelf() {
  return (uptr)pthread_self();
}

void FlushUnneededShadowMemory(uptr addr, uptr size) {
  madvise((void*)addr, size, MADV_DONTNEED);
}

void NoHugePagesInRegion(uptr addr, uptr size) {
#ifdef MADV_NOHUGEPAGE  // May not be defined on old systems.
  madvise((void *)addr, size, MADV_NOHUGEPAGE);
#endif  // MADV_NOHUGEPAGE
}

void DontDumpShadowMemory(uptr addr, uptr length) {
#ifdef MADV_DONTDUMP
  madvise((void *)addr, length, MADV_DONTDUMP);
#endif
}

static rlim_t getlim(int res) {
  rlimit rlim;
  CHECK_EQ(0, getrlimit(res, &rlim));
  return rlim.rlim_cur;
}

static void setlim(int res, rlim_t lim) {
  // The following magic is to prevent clang from replacing it with memset.
  volatile struct rlimit rlim;
  rlim.rlim_cur = lim;
  rlim.rlim_max = lim;
  if (setrlimit(res, const_cast<struct rlimit *>(&rlim))) {
    Report("ERROR: %s setrlimit() failed %d\n", SanitizerToolName, errno);
    Die();
  }
}

void DisableCoreDumperIfNecessary() {
  if (common_flags()->disable_coredump) {
    setlim(RLIMIT_CORE, 0);
  }
}

bool StackSizeIsUnlimited() {
  rlim_t stack_size = getlim(RLIMIT_STACK);
  return (stack_size == RLIM_INFINITY);
}

void SetStackSizeLimitInBytes(uptr limit) {
  setlim(RLIMIT_STACK, (rlim_t)limit);
  CHECK(!StackSizeIsUnlimited());
}

bool AddressSpaceIsUnlimited() {
  rlim_t as_size = getlim(RLIMIT_AS);
  return (as_size == RLIM_INFINITY);
}

void SetAddressSpaceUnlimited() {
  setlim(RLIMIT_AS, RLIM_INFINITY);
  CHECK(AddressSpaceIsUnlimited());
}

void SleepForSeconds(int seconds) {
  sleep(seconds);
}

void SleepForMillis(int millis) {
  usleep(millis * 1000);
}

void Abort() {
  abort();
}

int Atexit(void (*function)(void)) {
#ifndef SANITIZER_GO
  return atexit(function);
#else
  return 0;
#endif
}

int internal_isatty(fd_t fd) {
  return isatty(fd);
}

#ifndef SANITIZER_GO
// TODO(glider): different tools may require different altstack size.
static const uptr kAltStackSize = SIGSTKSZ * 4;  // SIGSTKSZ is not enough.

void SetAlternateSignalStack() {
  stack_t altstack, oldstack;
  CHECK_EQ(0, sigaltstack(0, &oldstack));
  // If the alternate stack is already in place, do nothing.
  // Android always sets an alternate stack, but it's too small for us.
  if (!SANITIZER_ANDROID && !(oldstack.ss_flags & SS_DISABLE)) return;
  // TODO(glider): the mapped stack should have the MAP_STACK flag in the
  // future. It is not required by man 2 sigaltstack now (they're using
  // malloc()).
  void* base = MmapOrDie(kAltStackSize, __func__);
  altstack.ss_sp = (char*) base;
  altstack.ss_flags = 0;
  altstack.ss_size = kAltStackSize;
  CHECK_EQ(0, sigaltstack(&altstack, 0));
}

void UnsetAlternateSignalStack() {
  stack_t altstack, oldstack;
  altstack.ss_sp = 0;
  altstack.ss_flags = SS_DISABLE;
  altstack.ss_size = kAltStackSize;  // Some sane value required on Darwin.
  CHECK_EQ(0, sigaltstack(&altstack, &oldstack));
  UnmapOrDie(oldstack.ss_sp, oldstack.ss_size);
}

typedef void (*sa_sigaction_t)(int, siginfo_t *, void *);
static void MaybeInstallSigaction(int signum,
                                  SignalHandlerType handler) {
  if (!IsDeadlySignal(signum))
    return;
  struct sigaction sigact;
  internal_memset(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = (sa_sigaction_t)handler;
  // Do not block the signal from being received in that signal's handler.
  // Clients are responsible for handling this correctly.
  sigact.sa_flags = SA_SIGINFO | SA_NODEFER;
  if (common_flags()->use_sigaltstack) sigact.sa_flags |= SA_ONSTACK;
  CHECK_EQ(0, internal_sigaction(signum, &sigact, 0));
  VReport(1, "Installed the sigaction for signal %d\n", signum);
}

void InstallDeadlySignalHandlers(SignalHandlerType handler) {
  // Set the alternate signal stack for the main thread.
  // This will cause SetAlternateSignalStack to be called twice, but the stack
  // will be actually set only once.
  if (common_flags()->use_sigaltstack) SetAlternateSignalStack();
  MaybeInstallSigaction(SIGSEGV, handler);
  MaybeInstallSigaction(SIGBUS, handler);
}
#endif  // SANITIZER_GO

bool IsAccessibleMemoryRange(uptr beg, uptr size) {
  uptr page_size = GetPageSizeCached();
  // Checking too large memory ranges is slow.
  CHECK_LT(size, page_size * 10);
  int sock_pair[2];
  if (pipe(sock_pair))
    return false;
  uptr bytes_written =
      internal_write(sock_pair[1], reinterpret_cast<void *>(beg), size);
  int write_errno;
  bool result;
  if (internal_iserror(bytes_written, &write_errno)) {
    CHECK_EQ(EFAULT, write_errno);
    result = false;
  } else {
    result = (bytes_written == size);
  }
  internal_close(sock_pair[0]);
  internal_close(sock_pair[1]);
  return result;
}

}  // namespace __sanitizer

#endif  // SANITIZER_POSIX
