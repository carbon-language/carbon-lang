// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/error.h"

namespace Carbon {

static const char UserErrorMessage[] = "ERROR: ";
static const char CompilationErrorMessage[] = "COMPILATION ERROR: ";
static const char RuntimeErrorMessage[] = "RUNTIME ERROR: ";

ErrorInternal::ExitingStream::~ExitingStream() {
  llvm::errs() << "\n";
  exit(-1);
}

ErrorInternal::ExitingStream FatalUserError(ErrorLine none) {
  llvm::errs() << UserErrorMessage;
  return ErrorInternal::ExitingStream();
}

ErrorInternal::ExitingStream FatalUserError(int line_num) {
  llvm::errs() << UserErrorMessage << line_num << ": ";
  return ErrorInternal::ExitingStream();
}

ErrorInternal::ExitingStream FatalCompilationError(ErrorLine none) {
  llvm::errs() << CompilationErrorMessage;
  return ErrorInternal::ExitingStream();
}

ErrorInternal::ExitingStream FatalCompilationError(int line_num) {
  llvm::errs() << CompilationErrorMessage << line_num << ": ";
  return ErrorInternal::ExitingStream();
}

ErrorInternal::ExitingStream FatalRuntimeError(ErrorLine none) {
  llvm::errs() << RuntimeErrorMessage;
  return ErrorInternal::ExitingStream();
}

ErrorInternal::ExitingStream FatalRuntimeError(int line_num) {
  llvm::errs() << RuntimeErrorMessage << line_num << ": ";
  return ErrorInternal::ExitingStream();
}

}  // namespace Carbon
