// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_ERROR_H_
#define EXECUTABLE_SEMANTICS_COMMON_ERROR_H_

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

enum class ErrorLine { None };

namespace ErrorInternal {
class ExitingStream {
  LLVM_ATTRIBUTE_NORETURN virtual ~ExitingStream() {
    // Finish with a newline.
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

// Prints an error and exits. This should be used for non-recoverable errors
// with user input.
//
// For example:
//   FatalUserError(line_num) << "Line is bad!";
//   FatalUserError(ErrorLine::None) << "Application is bad!";

class FatalUserError {
 public:
  FatalUserError(int line_num) {
    WritePrefix();
    llvm::errs() << line_num << ": ";
  }
  FatalUserError(ErrorLine no_line) { WritePrefix(); }
  LLVM_ATTRIBUTE_NORETURN virtual ~FatalUserError() {
    // Finish with a newline.
    llvm::errs() << "\n";
    exit(-1);
  }

  // Forward output to llvm::errs.
  template <typename T>
  FatalUserError& operator<<(const T& message) {
    llvm::errs() << message;
    return *this;
  }

 protected:
  virtual void WritePrefix() { llvm::errs() << "ERROR: "; }
};

class FatalRuntimeError : public FatalUserError {
 public:
  using FatalUserError::FatalUserError;

 protected:
  void WritePrefix() override { llvm::errs() << "RUNTIME ERROR: "; }
};

class FatalCompilationError : public FatalUserError {
 public:
  using FatalUserError::FatalUserError;

 protected:
  void WritePrefix() override { llvm::errs() << "COMPILATION ERROR: "; }
};

}  // namespace ErrorInternal

#endif  // EXECUTABLE_SEMANTICS_COMMON_ERROR_H_
