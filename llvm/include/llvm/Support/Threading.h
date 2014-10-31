//===-- llvm/Support/Threading.h - Control multithreading mode --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares helper functions for running LLVM in a multi-threaded
// environment.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_THREADING_H
#define LLVM_SUPPORT_THREADING_H

#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX

#if defined(LLVM_ON_UNIX)
#include <mutex>
#else
#include "llvm/Support/Atomic.h"
#endif

namespace llvm {
  /// Returns true if LLVM is compiled with support for multi-threading, and
  /// false otherwise.
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

#if defined(LLVM_ON_UNIX)
typedef std::once_flag once_flag;
#define LLVM_DEFINE_ONCE_FLAG(flag) static once_flag flag
#else
enum InitStatus {
  Done = -1,
  Uninitialized = 0,
  Wait = 1
};
typedef volatile sys::cas_flag once_flag;

#define LLVM_DEFINE_ONCE_FLAG(flag) static once_flag flag = Uninitialized
#endif

/// \brief Execute the function specified as a parameter once.
///
/// Typical usage:
/// \code
///   void foo() {...};
///   ...
///   LLVM_DEFINE_ONCE_FLAG(flag);
///   call_once(flag, foo);
/// \endcode
///
/// \param flag Flag used for tracking whether or not this has run.
/// \param UserFn Function to call once.
void call_once(once_flag &flag, void (*UserFn)(void));
}

#endif
