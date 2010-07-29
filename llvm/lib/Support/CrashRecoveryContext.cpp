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
#include "llvm/System/ThreadLocal.h"
#include <setjmp.h>
#include <cstdio>
using namespace llvm;

namespace {

struct CrashRecoveryContextImpl;

static sys::ThreadLocal<const CrashRecoveryContextImpl> CurrentContext;

struct CrashRecoveryContextImpl {
  std::string Backtrace;
  ::jmp_buf JumpBuffer;
  volatile unsigned Failed : 1;

public:
  CrashRecoveryContextImpl() : Failed(false) {
    CurrentContext.set(this);
  }
  ~CrashRecoveryContextImpl() {
    CurrentContext.set(0);
  }

  void HandleCrash() {
    assert(!Failed && "Crash recovery context already failed!");
    Failed = true;

    // FIXME: Stash the backtrace.

    // Jump back to the RunSafely we were called under.
    longjmp(JumpBuffer, 1);
  }
};

}

static bool gCrashRecoveryEnabled = false;

CrashRecoveryContext::~CrashRecoveryContext() {
  CrashRecoveryContextImpl *CRCI = (CrashRecoveryContextImpl *) Impl;
  delete CRCI;
}

#ifdef LLVM_ON_WIN32

// FIXME: No real Win32 implementation currently.

void CrashRecoveryContext::Enable() {
  if (gCrashRecoveryEnabled)
    return;

  gCrashRecoveryEnabled = true;
}

void CrashRecoveryContext::Disable() {
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

static struct {
  int Signal;
  struct sigaction PrevAction;
} SignalInfo[] = {
  { SIGABRT, {} },
  { SIGBUS,  {} },
  { SIGFPE,  {} },
  { SIGILL,  {} },
  { SIGSEGV, {} },
  { SIGTRAP, {} },
};
static const unsigned NumSignals = sizeof(SignalInfo) / sizeof(SignalInfo[0]);

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
  if (gCrashRecoveryEnabled)
    return;

  gCrashRecoveryEnabled = true;

  // Setup the signal handler.
  struct sigaction Handler;
  Handler.sa_handler = CrashRecoverySignalHandler;
  Handler.sa_flags = 0;
  sigemptyset(&Handler.sa_mask);

  for (unsigned i = 0; i != NumSignals; ++i) {
    sigaction(SignalInfo[i].Signal, &Handler,
              &SignalInfo[i].PrevAction);
  }
}

void CrashRecoveryContext::Disable() {
  if (!gCrashRecoveryEnabled)
    return;

  gCrashRecoveryEnabled = false;

  // Restore the previous signal handlers.
  for (unsigned i = 0; i != NumSignals; ++i)
    sigaction(SignalInfo[i].Signal, &SignalInfo[i].PrevAction, 0);
}

#endif

bool CrashRecoveryContext::RunSafely(void (*Fn)(void*), void *UserData) {
  // If crash recovery is disabled, do nothing.
  if (gCrashRecoveryEnabled) {
    assert(!Impl && "Crash recovery context already initialized!");
    CrashRecoveryContextImpl *CRCI = new CrashRecoveryContextImpl;
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
