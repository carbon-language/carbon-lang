// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_VLOG_H_
#define CARBON_COMMON_VLOG_H_

#include "common/ostream.h"
#include "common/template_string.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon::Internal {

// Implements verbose logging.
//
// This is designed to minimize the overhead in callers by being a
// forcibly-outlined routine that takes a minimal number of parameters.
//
// Internally uses `llvm::formatv` to render the format string with any value
// arguments, and streams the result to the provided stream.
template <TemplateString FormatStr, typename... Ts>
[[clang::noinline]] auto VLogImpl(llvm::raw_ostream* stream, Ts&&... values)
    -> void {
  *stream << llvm::formatv(FormatStr.c_str(), std::forward<Ts>(values)...);
}

}  // namespace Carbon::Internal

// Logs when verbose logging is enabled. CARBON_VLOG_TO uses a provided stream;
// CARBON_VLOG requires a member named `vlog_stream_`.
//
// For example:
//   CARBON_VLOG_TO(vlog_stream, "Verbose message: {0}", "extra information");
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
#define CARBON_VLOG_TO(Stream, FormatStr, ...)                         \
  __builtin_expect(Stream == nullptr, true)                            \
      ? (void)0                                                        \
      : Carbon::Internal::VLogImpl<"" FormatStr>(Stream __VA_OPT__(, ) \
                                                     __VA_ARGS__)

#define CARBON_VLOG(FormatStr, ...) \
  CARBON_VLOG_TO(vlog_stream_, FormatStr, __VA_ARGS__)

#endif  // CARBON_COMMON_VLOG_H_
