// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_CHECK_H_
#define EXECUTABLE_SEMANTICS_COMMON_CHECK_H_

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#define CHECK(condition)                                    \
  if (!(condition)) {                                       \
    llvm::sys::PrintStackTrace(llvm::errs());               \
    llvm::report_fatal_error("CHECK failure: " #condition); \
  }

#endif  // EXECUTABLE_SEMANTICS_COMMON_CHECK_H_
