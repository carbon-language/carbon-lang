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
//   CARBON_CHECK(is_valid, "Data is not valid!");
//
// The condition must be parenthesized if it contains top-level commas, for
// example in a template argument list:
//   CARBON_CHECK((inst.IsOneOf<Call, TupleLiteral>()),
//                "Unexpected inst {0}", inst);
#define CARBON_CHECK(condition, ...)         \
  CARBON_INTERNAL_CHECK_CONDITION(condition) \
  ? (void)0 : CARBON_INTERNAL_CHECK(condition __VA_OPT__(, ) __VA_ARGS__)

// DCHECK calls CHECK in debug mode, and does nothing otherwise.
#ifndef NDEBUG
#define CARBON_DCHECK(condition, ...) \
  CARBON_CHECK(condition __VA_OPT__(, ) __VA_ARGS__)
#else
// When in a debug build we want to preserve as much as we can of how the
// parameters are used, other than making them be trivially in dead code and
// eliminated by the optimizer. As a consequence we preserve the condition but
// prefix it with a short-circuit operator, and we still emit the (dead) call to
// the check implementation. But we use a special implementation that reduces
// the compile time cost.
#define CARBON_DCHECK(condition, ...)                  \
  (true || CARBON_INTERNAL_CHECK_CONDITION(condition)) \
      ? (void)0                                        \
      : CARBON_INTERNAL_DEAD_DCHECK(condition __VA_OPT__(, ) __VA_ARGS__)
#endif

// This is similar to CHECK, but is unconditional. Writing
// `CARBON_FATAL("message")` is clearer than `CARBON_CHECK(false, "message")
// because it avoids confusion about control flow.
//
// For example:
//   CARBON_FATAL("Unreachable!");
#define CARBON_FATAL(...) CARBON_INTERNAL_FATAL(__VA_ARGS__)

}  // namespace Carbon

#endif  // CARBON_COMMON_CHECK_H_
