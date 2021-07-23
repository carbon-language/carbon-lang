// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/error.h"

namespace Carbon {

ErrorInternal::ExitingStream FatalUserError(ErrorLine none) {
  auto stream = ErrorInternal::ExitingStream();
  stream << "ERROR: ";
  return stream;
}

ErrorInternal::ExitingStream FatalUserError(int line_num) {
  auto stream = FatalUserError(ErrorLine::None);
  stream << line_num << ": ";
  return stream;
}

ErrorInternal::ExitingStream FatalCompilationError(ErrorLine none) {
  auto stream = ErrorInternal::ExitingStream();
  stream << "COMPILATION ERROR: ";
  return stream;
}

ErrorInternal::ExitingStream FatalCompilationError(int line_num) {
  auto stream = FatalCompilationError(ErrorLine::None);
  stream << line_num << ": ";
  return stream;
}

ErrorInternal::ExitingStream FatalRuntimeError(ErrorLine none) {
  auto stream = ErrorInternal::ExitingStream();
  stream << "RUNTIME ERROR: ";
  return stream;
}

ErrorInternal::ExitingStream FatalRuntimeError(int line_num) {
  auto stream = FatalRuntimeError(ErrorLine::None);
  stream << line_num << ": ";
  return stream;
}

}  // namespace Carbon
