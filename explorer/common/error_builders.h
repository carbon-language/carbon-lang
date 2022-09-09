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
//   return ProgramError(line_num) << "Line is bad!";
//   return ProgramError() << "Application is bad!";
//
// Where possible, try to identify the error as a compilation or runtime error.
// Use CHECK/FATAL for internal errors. The generic program error option is
// provided as a fallback for cases that don't fit those classifications.

inline auto CompilationError(SourceLocation loc) -> ErrorBuilder {
  return ErrorBuilder("COMPILATION ERROR", loc.ToString());
}

inline auto ProgramError(SourceLocation loc) -> ErrorBuilder {
  return ErrorBuilder("PROGRAM ERROR", loc.ToString());
}

inline auto RuntimeError(SourceLocation loc) -> ErrorBuilder {
  return ErrorBuilder("RUNTIME ERROR", loc.ToString());
}

}  // namespace Carbon

#endif  // CARBON_EXPLORER_COMMON_ERROR_BUILDERS_H_
