//===-- Support/Lock.cpp - Platform-agnostic mutual exclusion -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Implementation of various methods supporting platform-agnostic lock
// abstraction. See Support/Lock.h for details.
//
//===----------------------------------------------------------------------===//

#include "Support/Lock.h"

using namespace llvm;

Lock Lock::create () {
  // Currently we only support creating POSIX pthread_mutex_t locks.
  // In the future we might want to construct different kinds of locks
  // based on what OS is running.
  return POSIXLock ();
}
