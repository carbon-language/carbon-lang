// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_CHECK_H_
#define COMMON_CHECK_H_

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// Wraps a stream and exiting for fatal errors.
class ExitingStream {
 public:
  LLVM_ATTRIBUTE_NORETURN ~ExitingStream() {
    // Finish with a newline.
    llvm::errs() << "\n";
    exit(-1);
  }

  // Indicates that initial input is in, so this is where a ": " should be added
  // before user input.
  ExitingStream& add_separator() {
    separator = true;
    return *this;
  }

  // Prints a stack traces.
  ExitingStream& print_stack() {
    separator = true;
    return *this;
  }

  // If the bool cast occurs, it's because the condition is false. This supports
  // && short-circuiting the creation of ExitingStream.
  explicit operator bool() const { return true; }

  // Forward output to llvm::errs.
  template <typename T>
  ExitingStream& operator<<(const T& message) {
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

// Checks the given condition, and if it's false, prints a stack, streams the
// error message, then exits. This should be used for unexpected errors, such as
// a bug in the application.
//
// For example:
//   CHECK(is_valid) << "Data is not valid!";
#define CHECK(condition)                                                      \
  (!(condition)) && (Carbon::ExitingStream() << "CHECK failure: " #condition) \
                        .add_separator()

// This is similar to CHECK, but is unconditional.
//
// For example:
//   FATAL() << "Unreachable!";
#define FATAL() Carbon::ExitingStream() << "FATAL: "

}  // namespace Carbon

#endif  // COMMON_CHECK_H_
