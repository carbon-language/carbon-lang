// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_CHECK_H_
#define COMMON_CHECK_H_

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace CheckInternal {

// Wraps a stream and exiting for CHECK.
class ExitWrapper {
 public:
  ExitWrapper() {}
  ~ExitWrapper() {
    // Finish with a newline.
    llvm::errs() << "\n";
    exit(-1);
  }

  // Indicates that initial input is in, so this is where a ": " should be added
  // before user input.
  ExitWrapper& AddSeparator() {
    separator = true;
    return *this;
  }

  // Prints the current stack trace.
  ExitWrapper& PrintStackTrace() {
    llvm::sys::PrintStackTrace(llvm::errs());
    return *this;
  }

  // If the bool cast occurs, it's because the condition is false. This supports
  // && short-circuiting the creation of ExitWrapper.
  explicit operator bool() const { return true; }

  // Forward output to llvm::errs.
  template <typename T>
  ExitWrapper& operator<<(const T& message) {
    if (separator) {
      llvm::errs() << ": ";
      separator = false;
    }
    llvm::errs() << message;
    return *this;
  }

 private:
  // Whether a separator should be printed if << is used again.
  bool separator = false;
};

}  // namespace CheckInternal

// Checks the given condition, and if it's **false**, prints the stack, an
// error, and exits.
//
// For example:
//   CHECK(is_valid) << "Data is not valid!";
// Would print:
//   <stack trace>
//   CHECK failure: is_valid: Data is not valid!
#define CHECK(condition)                                            \
  (!(condition)) && (CheckInternal::ExitWrapper().PrintStackTrace() \
                     << "CHECK failure: " #condition)               \
                        .AddSeparator()

// Checks the given condition, and if it's **true**, prints an error and exits.
//
// For example:
//   FATAL_IF(is_invalid) << "Data is not valid!";
// Would print:
//   FATAL: Data is not valid!
#define FATAL_IF(condition) \
  ((condition)) && (CheckInternal::ExitWrapper() << "FATAL: ")

#endif  // COMMON_CHECK_H_
