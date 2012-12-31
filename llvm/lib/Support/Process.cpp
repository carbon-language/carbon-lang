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
#include "llvm/Support/Process.h"
#include "llvm/Support/ErrorHandling.h"

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

#if defined(_MSC_VER)
#pragma warning(pop)
#endif


//===----------------------------------------------------------------------===//
// Implementations of legacy functions in terms of the new self_process object.

unsigned Process::GetPageSize() {
  return process::get_self()->page_size();
}


// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Process.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Windows/Process.inc"
#endif
