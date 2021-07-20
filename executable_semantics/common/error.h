// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_ERROR_H_
#define EXECUTABLE_SEMANTICS_COMMON_ERROR_H_

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace ErrorInternal {

// Wraps a stream and exiting behaviors.
class ExitWrapper {
 public:
  ~ExitWrapper() {
    // Finish with a newline.
    llvm::errs() << "\n";
    exit(-1);
  }

  // If the bool cast occurs, it's because the condition is false. This supports
  // && short-circuiting the creation of ExitWrapper.
  explicit operator bool() const { return true; }

  // Forward output to llvm::errs.
  template <typename T>
  ExitWrapper& operator<<(const T& message) {
    llvm::errs() << message;
    return *this;
  }
};

}  // namespace ErrorInternal

// Checks the given condition, and if it's true, prints an error and exits.
// This should be used for non-recoverable errors with user input.
//
// For example:
//   USER_ERROR_IF(is_invalid) << "Data is not valid!";
// Would print:
//   FATAL: Data is not valid!
#define USER_ERROR_IF(condition) \
  ((condition)) && (ErrorInternal::ExitWrapper() << "ERROR: ")

#endif  // EXECUTABLE_SEMANTICS_COMMON_ERROR_H_
