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
#include <setjmp.h>
using namespace llvm;

namespace {

struct CrashRecoveryContextImpl;

struct CrashRecoveryContextImpl {
  std::string Backtrace;
  ::jmp_buf JumpBuffer;
  volatile unsigned Failed : 1;

public:
  CrashRecoveryContextImpl() : Failed(false) {}

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
