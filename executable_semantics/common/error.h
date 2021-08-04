// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_ERROR_H_
#define EXECUTABLE_SEMANTICS_COMMON_ERROR_H_

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

namespace ErrorInternal {

// An error-printing stream that exits on destruction.
class ExitingStream {
 public:
  // Ends the error with a newline and exits.
  LLVM_ATTRIBUTE_NORETURN virtual ~ExitingStream() {
    llvm::errs() << "\n";
    exit(-1);
  }

  // Forward output to llvm::errs.
  template <typename T>
  ExitingStream& operator<<(const T& message) {
    llvm::errs() << message;
    return *this;
  }
};

}  // namespace ErrorInternal

// Prints an error and exits. This should be used for non-recoverable errors
// with user input.
//
// For example:
//   FATAL_USER_ERROR(line_num) << "Line is bad!";
//   FATAL_USER_ERROR_NO_LINE() << "Application is bad!";
//
// Where possible, try to identify the error as a compilation error or runtime
// error. The generic user error option is provided as a fallback for cases that
// don't fit either of those classifications.

#define FATAL_USER_ERROR_NO_LINE() \
  Carbon::ErrorInternal::ExitingStream() << "ERROR: "

#define FATAL_USER_ERROR(line) FATAL_USER_ERROR_NO_LINE() << line << ": "

#define FATAL_COMPILATION_ERROR_NO_LINE() \
  Carbon::ErrorInternal::ExitingStream() << "COMPILATION ERROR: "

#define FATAL_COMPILATION_ERROR(line) \
  FATAL_COMPILATION_ERROR_NO_LINE() << line << ": "

#define FATAL_INTERNAL_ERROR_NO_LINE() \
  Carbon::ErrorInternal::ExitingStream() << "INTERNAL ERROR: "

#define FATAL_INTERNAL_ERROR(line) \
  FATAL_INTERNAL_ERROR_NO_LINE() << line << ": "

#define FATAL_RUNTIME_ERROR_NO_LINE() \
  Carbon::ErrorInternal::ExitingStream() << "RUNTIME ERROR: "

#define FATAL_RUNTIME_ERROR(line) FATAL_RUNTIME_ERROR_NO_LINE() << line << ": "

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_ERROR_H_
