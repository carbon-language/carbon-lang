// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_CHECK_H_
#define CARBON_COMMON_CHECK_H_

#include "common/check_internal.h"

namespace Carbon {

// Checks the given condition, and if it's false, prints a stack, streams the
// error message, then exits. This should be used for unexpected errors, such as
// a bug in the application.
//
// For example:
//   CARBON_CHECK(is_valid) << "Data is not valid!";
#define CARBON_CHECK(...)                                                   \
  (__VA_ARGS__) ? (void)0                                                   \
                : CARBON_CHECK_INTERNAL_STREAM()                            \
                      << "CHECK failure at " << __FILE__ << ":" << __LINE__ \
                      << ": " #__VA_ARGS__                                  \
                      << Carbon::Internal::ExitingStream::AddSeparator()

// DCHECK calls CHECK in debug mode, and does nothing otherwise.
#ifndef NDEBUG
#define CARBON_DCHECK(...) CARBON_CHECK(__VA_ARGS__)
#else
#define CARBON_DCHECK(...) CARBON_CHECK(true || (__VA_ARGS__))
#endif

// This is similar to CHECK, but is unconditional. Writing CARBON_FATAL() is
// clearer than CARBON_CHECK(false) because it avoids confusion about control
// flow.
//
// For example:
//   CARBON_FATAL() << "Unreachable!";
#define CARBON_FATAL()           \
  CARBON_CHECK_INTERNAL_STREAM() \
      << "FATAL failure at " << __FILE__ << ":" << __LINE__ << ": "

}  // namespace Carbon

#endif  // CARBON_COMMON_CHECK_H_
