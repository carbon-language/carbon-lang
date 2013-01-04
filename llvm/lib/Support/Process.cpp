//===-- Process.cpp - Implement OS Process Concept --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file implements the operating system Process concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Process.h"

using namespace llvm;
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

// Empty virtual destructor to anchor the vtable for the process class.
process::~process() {}

self_process *process::get_self() {
  // Use a function local static for thread safe initialization and allocate it
  // as a raw pointer to ensure it is never destroyed.
  static self_process *SP = new self_process();

  return SP;
}

#if defined(_MSC_VER)
// Visual Studio complains that the self_process destructor never exits. This
// doesn't make much sense, as that's the whole point of calling abort... Just
// silence this warning.
#pragma warning(push)
#pragma warning(disable:4722)
#endif

// The destructor for the self_process subclass must never actually be
// executed. There should be at most one instance of this class, and that
// instance should live until the process terminates to avoid the potential for
// racy accesses during shutdown.
self_process::~self_process() {
  llvm_unreachable("This destructor must never be executed!");
}

/// \brief A helper function to compute the elapsed wall-time since the program
/// started.
///
/// Note that this routine actually computes the elapsed wall time since the
/// first time it was called. However, we arrange to have it called during the
/// startup of the process to get approximately correct results.
static TimeValue getElapsedWallTime() {
  static TimeValue &StartTime = *new TimeValue(TimeValue::now());
  return TimeValue::now() - StartTime;
}

/// \brief A special global variable to ensure we call \c getElapsedWallTime
/// during global initialization of the program.
///
/// Note that this variable is never referenced elsewhere. Doing so could
/// create race conditions during program startup or shutdown.
static volatile TimeValue DummyTimeValue = getElapsedWallTime();

// Implement this routine by using the static helpers above. They're already
// portable.
TimeValue self_process::get_wall_time() const {
  return getElapsedWallTime();
}


#if defined(_MSC_VER)
#pragma warning(pop)
#endif


// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Process.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Windows/Process.inc"
#endif
