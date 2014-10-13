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

#if !defined(__MINGW__)
#include <mutex>
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

/// \brief Execute the function specified as a template parameter once.
///
/// Calls \p UserFn once ever. The call uniqueness is based on the address of
/// the function passed in via the template arguement. This means no matter how
/// many times you call llvm_call_once<foo>() in the same or different
/// locations, foo will only be called once.
///
/// Typical usage:
/// \code
///   void foo() {...};
///   ...
///   llvm_call_once<foo>();
/// \endcode
///
/// \param UserFn Function to call once.
template <void (*UserFn)(void)> void llvm_call_once() {
#if !defined(__MINGW__)
  static std::once_flag flag;
  std::call_once(flag, UserFn);
#else
  struct InitOnceWrapper {
    InitOnceWrapper() { UserFn(); }
  };
  static InitOnceWrapper InitOnceVar;
#endif
}
}

#endif
