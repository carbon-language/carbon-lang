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

namespace llvm {
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

// The destructor for the self_process subclass must never actually be
// executed. There should be at most one instance of this class, and that
// instance should live until the process terminates to avoid the potential for
// racy accesses during shutdown.
self_process::~self_process() {
  llvm_unreachable("This destructor must never be executed!");
}

}

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Process.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Windows/Process.inc"
#endif
