// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_VLOG_INTERNAL_H_
#define CARBON_COMMON_VLOG_INTERNAL_H_

#include "common/ostream.h"
#include "common/template_string.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon::Internal {

// Wraps a stream and exiting for fatal errors. Should only be used by check.h
// macros.
//
// TODO: Remove this when the last streaming `vlog` is replaced with a function
// call variant.
class VLoggingStream {
 public:
  // Internal type used in macros to dispatch to the `operator|` overload.
  struct Helper {};

  explicit VLoggingStream(llvm::raw_ostream* stream)
      // Prefix the buffer with the current bug report message.
      : stream_(stream) {}

  ~VLoggingStream() = default;

  // If the bool cast occurs, it's because the condition is false. This supports
  // && short-circuiting the creation of ExitingStream.
  explicit operator bool() const { return true; }

  // Forward output to llvm::errs.
  template <typename T>
  auto operator<<(const T& message) -> VLoggingStream& {
    *stream_ << message;
    return *this;
  }

  // Low-precedence binary operator overload used in vlog.h macros.
  friend auto operator|(Helper /*helper*/, VLoggingStream& /*stream*/) -> void {
  }

 private:
  llvm::raw_ostream* stream_;
};

// Implements verbose logging.
//
// This is designed to minimize the overhead in callers by being a
// forcibly-outlined routine that takes a minimal number of parameters.
//
// Internally uses `llvm::formatv` to render the format string with any value
// arguments, and streams the result to the provided stream.
template <TemplateString FormatStr, typename... Ts>
[[clang::noinline, clang::preserve_most]] auto VLogImpl(
    llvm::raw_ostream* stream, Ts&&... values) -> void {
  *stream << llvm::formatv(FormatStr.c_str(), std::forward<Ts>(values)...);
}

}  // namespace Carbon::Internal

// Raw logging stream. This should be used when building the streaming forms of
// vlog macros.
#define CARBON_VLOG_INTERNAL(stream)           \
  Carbon::Internal::VLoggingStream::Helper() | \
      Carbon::Internal::VLoggingStream(stream)

// Raw logging call. This should be used when building the format-string forms
// of vlog macros.
#define CARBON_VLOG_INTERNAL_CALL(stream, FormatStr, ...) \
  Carbon::Internal::VLogImpl<"" FormatStr>(stream __VA_OPT__(, ) __VA_ARGS__)

#endif  // CARBON_COMMON_VLOG_INTERNAL_H_
