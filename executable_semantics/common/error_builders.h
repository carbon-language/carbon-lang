// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_ERROR_BUILDERS_H_
#define EXECUTABLE_SEMANTICS_COMMON_ERROR_BUILDERS_H_

#include <optional>

#include "common/check.h"
#include "common/error.h"
#include "executable_semantics/common/source_location.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// Builds an Error instance with the specified message. This should be used for
// non-recoverable errors with user input.
//
// For example:
//   return ProgramErrorBuilder(line_num) << "Line is bad!";
//   return ProgramErrorBuilder() << "Application is bad!";
//
// Where possible, try to identify the error as a compilation or
// runtime error. Use CHECK/FATAL for internal errors.

class CompilationErrorBuilder : public ErrorBuilder {
 public:
  explicit CompilationErrorBuilder(SourceLocation loc) {
    (void)(*this << "COMPILATION ERROR: " << loc << ": ");
  }
};

class ProgramErrorBuilder : public ErrorBuilder {
 public:
  explicit ProgramErrorBuilder(SourceLocation loc) {
    (void)(*this << "PROGRAM ERROR: " << loc << ": ");
  }
};

class RuntimeErrorBuilder : public ErrorBuilder {
 public:
  explicit RuntimeErrorBuilder(SourceLocation loc) {
    (void)(*this << "RUNTIME ERROR: " << loc << ": ");
  }
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_ERROR_BUILDERS_H_
