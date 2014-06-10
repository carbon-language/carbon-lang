//===-- llvm/Support/Threading.h - Control multithreading mode --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TThis file defines llvm_start_multithreaded() and friends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_THREADING_H
#define LLVM_SUPPORT_THREADING_H

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"
#include <mutex>

namespace llvm {

#if LLVM_ENABLE_THREADS != 0
  typedef std::mutex mutex;
  typedef std::recursive_mutex recursive_mutex;
#else
  class null_mutex {
  public:
    void lock() { }
    void unlock() { }
    bool try_lock() { return true; }
  };
  typedef null_mutex mutex;
  typedef null_mutex recursive_mutex;
#endif

  /// llvm_get_global_lock() - returns the llvm global lock object.
  llvm::recursive_mutex &llvm_get_global_lock();

  /// llvm_is_multithreaded() - returns true if LLVM is compiled with support
  /// for multiple threads, and false otherwise.
  bool llvm_is_multithreaded();

  /// llvm_execute_on_thread - Execute the given \p UserFn on a separate
  /// thread, passing it the provided \p UserData.
  ///
  /// This function does not guarantee that the code will actually be executed
  /// on a separate thread or honoring the requested stack size, but tries to do
  /// so where system support is available.
  ///
  /// \param UserFn - The callback to execute.
  /// \param UserData - An argument to pass to the callback function.
  /// \param RequestedStackSize - If non-zero, a requested size (in bytes) for
  /// the thread stack.
  void llvm_execute_on_thread(void (*UserFn)(void*), void *UserData,
                              unsigned RequestedStackSize = 0);
}

#endif
