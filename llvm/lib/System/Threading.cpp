//===-- llvm/System/Threading.cpp- Control multithreading mode --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements llvm_start_multithreaded() and friends.
//
//===----------------------------------------------------------------------===//

#include "llvm/System/Threading.h"
#include "llvm/System/Atomic.h"
#include "llvm/System/Mutex.h"
#include "llvm/Config/config.h"
#include <cassert>

using namespace llvm;

static bool multithreaded_mode = false;

static sys::Mutex* global_lock = 0;

bool llvm::llvm_start_multithreaded() {
#ifdef LLVM_MULTITHREADED
  assert(!multithreaded_mode && "Already multithreaded!");
  multithreaded_mode = true;
  global_lock = new sys::Mutex(true);
  
  // We fence here to ensure that all initialization is complete BEFORE we
  // return from llvm_start_multithreaded().
  sys::MemoryFence();
  return true;
#else
  return false;
#endif
}

void llvm::llvm_stop_multithreaded() {
#ifdef LLVM_MULTITHREADED
  assert(multithreaded_mode && "Not currently multithreaded!");
  
  // We fence here to insure that all threaded operations are complete BEFORE we
  // return from llvm_stop_multithreaded().
  sys::MemoryFence();
  
  multithreaded_mode = false;
  delete global_lock;
#endif
}

bool llvm::llvm_is_multithreaded() {
  return multithreaded_mode;
}

void llvm::llvm_acquire_global_lock() {
  if (multithreaded_mode) global_lock->acquire();
}

void llvm::llvm_release_global_lock() {
  if (multithreaded_mode) global_lock->release();
}
