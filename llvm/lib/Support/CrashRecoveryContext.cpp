//===--- CrashRecoveryContext.cpp - Crash Recovery ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/ThreadLocal.h"
#include <setjmp.h>
#include <cstdio>
using namespace llvm;

namespace {

struct CrashRecoveryContextImpl;

static sys::ThreadLocal<const CrashRecoveryContextImpl> CurrentContext;

struct CrashRecoveryContextImpl {
  CrashRecoveryContext *CRC;
  std::string Backtrace;
  ::jmp_buf JumpBuffer;
  volatile unsigned Failed : 1;

public:
  CrashRecoveryContextImpl(CrashRecoveryContext *CRC) : CRC(CRC),
                                                        Failed(false) {
    CurrentContext.set(this);
  }
  ~CrashRecoveryContextImpl() {
    CurrentContext.erase();
  }

  void HandleCrash() {
    // Eliminate the current context entry, to avoid re-entering in case the
    // cleanup code crashes.
    CurrentContext.erase();

    assert(!Failed && "Crash recovery context already failed!");
    Failed = true;

    // FIXME: Stash the backtrace.

    // Jump back to the RunSafely we were called under.
    longjmp(JumpBuffer, 1);
  }
};

}

static sys::Mutex gCrashRecoveryContexMutex;
static bool gCrashRecoveryEnabled = false;

CrashRecoveryContext::~CrashRecoveryContext() {
  CrashRecoveryContextImpl *CRCI = (CrashRecoveryContextImpl *) Impl;
  delete CRCI;
}

CrashRecoveryContext *CrashRecoveryContext::GetCurrent() {
  const CrashRecoveryContextImpl *CRCI = CurrentContext.get();
  if (!CRCI)
    return 0;

  return CRCI->CRC;
}

#ifdef LLVM_ON_WIN32

// FIXME: No real Win32 implementation currently.

void CrashRecoveryContext::Enable() {
  sys::ScopedLock L(gCrashRecoveryContexMutex);

  if (gCrashRecoveryEnabled)
    return;

  gCrashRecoveryEnabled = true;
}

void CrashRecoveryContext::Disable() {
  sys::ScopedLock L(gCrashRecoveryContexMutex);

  if (!gCrashRecoveryEnabled)
    return;

  gCrashRecoveryEnabled = false;
}

#else

// Generic POSIX implementation.
//
// This implementation relies on synchronous signals being delivered to the
// current thread. We use a thread local object to keep track of the active
// crash recovery context, and install signal handlers to invoke HandleCrash on
// the active object.
//
// This implementation does not to attempt to chain signal handlers in any
// reliable fashion -- if we get a signal outside of a crash recovery context we
// simply disable crash recovery and raise the signal again.

#include <signal.h>

static int Signals[] = { SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGSEGV, SIGTRAP };
static const unsigned NumSignals = sizeof(Signals) / sizeof(Signals[0]);
static struct sigaction PrevActions[NumSignals];

static void CrashRecoverySignalHandler(int Signal) {
  // Lookup the current thread local recovery object.
  const CrashRecoveryContextImpl *CRCI = CurrentContext.get();

  if (!CRCI) {
    // We didn't find a crash recovery context -- this means either we got a
    // signal on a thread we didn't expect it on, the application got a signal
    // outside of a crash recovery context, or something else went horribly
    // wrong.
    //
    // Disable crash recovery and raise the signal again. The assumption here is
    // that the enclosing application will terminate soon, and we won't want to
    // attempt crash recovery again.
    //
    // This call of Disable isn't thread safe, but it doesn't actually matter.
    CrashRecoveryContext::Disable();
    raise(Signal);

    // The signal will be thrown once the signal mask is restored.
    return;
  }

  // Unblock the signal we received.
  sigset_t SigMask;
  sigemptyset(&SigMask);
  sigaddset(&SigMask, Signal);
  sigprocmask(SIG_UNBLOCK, &SigMask, 0);

  if (CRCI)
    const_cast<CrashRecoveryContextImpl*>(CRCI)->HandleCrash();
}

void CrashRecoveryContext::Enable() {
  sys::ScopedLock L(gCrashRecoveryContexMutex);

  if (gCrashRecoveryEnabled)
    return;

  gCrashRecoveryEnabled = true;

  // Setup the signal handler.
  struct sigaction Handler;
  Handler.sa_handler = CrashRecoverySignalHandler;
  Handler.sa_flags = 0;
  sigemptyset(&Handler.sa_mask);

  for (unsigned i = 0; i != NumSignals; ++i) {
    sigaction(Signals[i], &Handler, &PrevActions[i]);
  }
}

void CrashRecoveryContext::Disable() {
  sys::ScopedLock L(gCrashRecoveryContexMutex);

  if (!gCrashRecoveryEnabled)
    return;

  gCrashRecoveryEnabled = false;

  // Restore the previous signal handlers.
  for (unsigned i = 0; i != NumSignals; ++i)
    sigaction(Signals[i], &PrevActions[i], 0);
}

#endif

bool CrashRecoveryContext::RunSafely(void (*Fn)(void*), void *UserData) {
  // If crash recovery is disabled, do nothing.
  if (gCrashRecoveryEnabled) {
    assert(!Impl && "Crash recovery context already initialized!");
    CrashRecoveryContextImpl *CRCI = new CrashRecoveryContextImpl(this);
    Impl = CRCI;

    if (setjmp(CRCI->JumpBuffer) != 0) {
      return false;
    }
  }

  Fn(UserData);
  return true;
}

void CrashRecoveryContext::HandleCrash() {
  CrashRecoveryContextImpl *CRCI = (CrashRecoveryContextImpl *) Impl;
  assert(CRCI && "Crash recovery context never initialized!");
  CRCI->HandleCrash();
}

const std::string &CrashRecoveryContext::getBacktrace() const {
  CrashRecoveryContextImpl *CRC = (CrashRecoveryContextImpl *) Impl;
  assert(CRC && "Crash recovery context never initialized!");
  assert(CRC->Failed && "No crash was detected!");
  return CRC->Backtrace;
}

//

namespace {
struct RunSafelyOnThreadInfo {
  void (*UserFn)(void*);
  void *UserData;
  CrashRecoveryContext *CRC;
  bool Result;
};
}

static void RunSafelyOnThread_Dispatch(void *UserData) {
  RunSafelyOnThreadInfo *Info =
    reinterpret_cast<RunSafelyOnThreadInfo*>(UserData);
  Info->Result = Info->CRC->RunSafely(Info->UserFn, Info->UserData);
}
bool CrashRecoveryContext::RunSafelyOnThread(void (*Fn)(void*), void *UserData,
                                             unsigned RequestedStackSize) {
  RunSafelyOnThreadInfo Info = { Fn, UserData, this, false };
  llvm_execute_on_thread(RunSafelyOnThread_Dispatch, &Info, RequestedStackSize);
  return Info.Result;
}
