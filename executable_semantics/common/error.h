// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_ERROR_H_
#define EXECUTABLE_SEMANTICS_COMMON_ERROR_H_

#include "common/check.h"

namespace Carbon {

// Prints an error and exits. This should be used for non-recoverable errors
// with user input.
//
// For example:
//   FATAL_PROGRAM_ERROR(line_num) << "Line is bad!";
//   FATAL_PROGRAM_ERROR_NO_LINE() << "Application is bad!";
//
// Where possible, try to identify the error as a compilation or
// runtime error. Use CHECK/FATAL for internal errors. The generic program error
// option is provided as a fallback for cases that don't fit those
// classifications.

#define FATAL_PROGRAM_ERROR_NO_LINE() RAW_EXITING_STREAM() << "PROGRAM ERROR: "

#define FATAL_PROGRAM_ERROR(line) \
  FATAL_PROGRAM_ERROR_NO_LINE() << (line) << ": "

#define FATAL_COMPILATION_ERROR_NO_LINE() \
  RAW_EXITING_STREAM() << "COMPILATION ERROR: "

#define FATAL_COMPILATION_ERROR(line) \
  FATAL_COMPILATION_ERROR_NO_LINE() << (line) << ": "

#define FATAL_RUNTIME_ERROR_NO_LINE() RAW_EXITING_STREAM() << "RUNTIME ERROR: "

#define FATAL_RUNTIME_ERROR(line) \
  FATAL_RUNTIME_ERROR_NO_LINE() << (line) << ": "

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_ERROR_H_
