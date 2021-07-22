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

// Indicates there is no line for an error. Otherwise, the integer line is
// passed.
enum class ErrorLine { None };

// Prints an error and exits. This should be used for non-recoverable errors
// with user input.
//
// For example:
//   FatalUserError(line_num) << "Line is bad!";
//   FatalUserError(ErrorLine::None) << "Application is bad!";
ErrorInternal::ExitingStream FatalUserError(ErrorLine none);
ErrorInternal::ExitingStream FatalUserError(int line_num);
ErrorInternal::ExitingStream FatalCompilationError(ErrorLine none);
ErrorInternal::ExitingStream FatalCompilationError(int line_num);
ErrorInternal::ExitingStream FatalRuntimeError(ErrorLine none);
ErrorInternal::ExitingStream FatalRuntimeError(int line_num);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_ERROR_H_
