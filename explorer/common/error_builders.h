// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_COMMON_ERROR_BUILDERS_H_
#define CARBON_EXPLORER_COMMON_ERROR_BUILDERS_H_

#include "common/error.h"
#include "explorer/common/source_location.h"

namespace Carbon {

// Builds an Error instance with the specified message. This should be used
// for errors in the user-supplied Carbon code is incorrect. Use CHECK/FATAL
// instead for errors that indicate bugs in Explorer itself.
//
// For example:
//   return ProgramError(line_num) << "Line is bad!";

inline auto ProgramError(SourceLocation loc) -> ErrorBuilder {
  return ErrorBuilder(loc.ToString());
}

}  // namespace Carbon

#endif  // CARBON_EXPLORER_COMMON_ERROR_BUILDERS_H_
