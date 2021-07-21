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
  ExitWrapper() {
    // Start by printing a stack trace.
    llvm::sys::PrintStackTrace(llvm::errs());
  }
  ~ExitWrapper() {
    // Finish with a newline.
    llvm::errs() << "\n";
    exit(-1);
  }

  // Indicates that initial input is in, so this is where a ": " should be added
  // before user input.
  ExitWrapper& add_separator() {
    separator = true;
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

// Checks the given condition, and if it's false, prints a stack, streams the
// error message, then exits. This should be used for unexpected errors, such as
// a bug in the application.
//
// For example:
//   CHECK(is_valid) << "Data is not valid!";
#define CHECK(condition)                                             \
  (!(condition)) &&                                                  \
      (CheckInternal::ExitWrapper() << "CHECK failure: " #condition) \
          .add_separator()

#endif  // COMMON_CHECK_H_
