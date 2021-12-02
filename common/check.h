// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_CHECK_H_
#define COMMON_CHECK_H_

#include "common/check_internal.h"

namespace Carbon {

// Raw exiting stream. This should be used when building other forms of exiting
// macros like those below. It evaluates to a temporary `ExitingStream` object
// that can be manipulated, streamed into, and then will exit the program.
#define RAW_EXITING_STREAM() \
  Carbon::Internal::ExitingStream::Helper() | Carbon::Internal::ExitingStream()

// Checks the given condition, and if it's false, prints a stack, streams the
// error message, then exits. This should be used for unexpected errors, such as
// a bug in the application.
//
// For example:
//   CHECK(is_valid) << "Data is not valid!";
#define CHECK(condition)                                                  \
  (condition) ? (void)0                                                   \
              : RAW_EXITING_STREAM().TreatAsBug()                         \
                    << "CHECK failure at " << __FILE__ << ":" << __LINE__ \
                    << ": " #condition                                    \
                    << Carbon::Internal::ExitingStream::AddSeparator()

// This is similar to CHECK, but is unconditional. Writing FATAL() is clearer
// than CHECK(false) because it avoids confusion about control flow.
//
// For example:
//   FATAL() << "Unreachable!";
#define FATAL()                     \
  RAW_EXITING_STREAM().TreatAsBug() \
      << "FATAL failure at " << __FILE__ << ":" << __LINE__ << ": "

}  // namespace Carbon

#endif  // COMMON_CHECK_H_
