// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_COMMON_ERROR_BUILDERS_H_
#define CARBON_EXPLORER_COMMON_ERROR_BUILDERS_H_

#include "common/error.h"
#include "explorer/common/source_location.h"

namespace Carbon {

// Builds an Error instance with the specified message. This should be used for
// non-recoverable errors with user input.
//
// For example:
//   return CompilationError(line_num) << "Line is bad!";
//
// These should only be used to report errors in the user's Carbon code.
// Use CHECK/FATAL for errors that indicate bugs in the Explorer.

// Reports a compile-time error in the user's Carbon code.
inline auto CompilationError(SourceLocation loc) -> ErrorBuilder {
  return ErrorBuilder("COMPILATION ERROR", loc.ToString());
}

// Reports a run-time error in the user's Carbon code.
inline auto RuntimeError(SourceLocation loc) -> ErrorBuilder {
  return ErrorBuilder("RUNTIME ERROR", loc.ToString());
}

// Reports an error in the user's Carbon code that we aren't able to
// classify as compile-time or run-time. Prefer to use CompilationError or
// RuntimeError instead, if possible.
inline auto ProgramError(SourceLocation loc) -> ErrorBuilder {
  return ErrorBuilder("PROGRAM ERROR", loc.ToString());
}

}  // namespace Carbon

#endif  // CARBON_EXPLORER_COMMON_ERROR_BUILDERS_H_
