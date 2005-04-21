//===- llvm/System/Signals.h - Signal Handling support ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// unix signals occuring while your program is running.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_SIGNALS_H
#define LLVM_SYSTEM_SIGNALS_H

#include "llvm/System/Path.h"

namespace llvm {
namespace sys {

  /// This function registers signal handlers to ensure that if a signal gets
  /// delivered that the named file is removed.
  /// @brief Remove a file if a fatal signal occurs.
  void RemoveFileOnSignal(const Path &Filename);

  /// This function registers a signal handler to ensure that if a fatal signal
  /// gets delivered to the process that the named directory and all its
  /// contents are removed.
  /// @brief Remove a directory if a fatal signal occurs.
  void RemoveDirectoryOnSignal(const Path& path);

  /// When an error signal (such as SIBABRT or SIGSEGV) is delivered to the
  /// process, print a stack trace and then exit.
  /// @brief Print a stack trace if a fatal signal occurs.
  void PrintStackTraceOnErrorSignal();

} // End sys namespace
} // End llvm namespace

#endif
