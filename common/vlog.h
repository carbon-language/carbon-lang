// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_VLOG_H_
#define CARBON_COMMON_VLOG_H_

#include "common/vlog_internal.h"

namespace Carbon {

// Logs when verbose logging is enabled (vlog_stream_ is non-null).
//
// For example:
//   CARBON_VLOG("Verbose message: {0}", "extra information");
//
// The first argument must be a string literal format string valid for passing
// to `llvm::formatv`. If it contains any substitutions, those should be passed
// as subsequent arguments.
//
// Also supports a legacy syntax where no arguments are passed and the desired
// logging is streamed into the call:
//   CARBON_VLOG() << "Legacy verbose message";
//
// However, the streaming syntax has higher overhead and can inhibit inlining.
// Code should prefer the format string form, and eventually when all code has
// migrated the streaming interface will be removed.
#define CARBON_VLOG(...)                                                    \
  __builtin_expect(vlog_stream_ == nullptr, true)                           \
      ? (void)0                                                             \
      : CARBON_VLOG_INTERNAL##__VA_OPT__(_CALL)(vlog_stream_ __VA_OPT__(, ) \
                                                    __VA_ARGS__)

}  // namespace Carbon

#endif  // CARBON_COMMON_VLOG_H_
