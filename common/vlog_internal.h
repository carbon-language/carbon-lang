// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_VLOG_INTERNAL_H_
#define CARBON_COMMON_VLOG_INTERNAL_H_

#include <cstdlib>

#include "llvm/Support/raw_ostream.h"

namespace Carbon::Internal {

// Wraps a stream and exiting for fatal errors. Should only be used by check.h
// macros.
class VLoggingStream {
 public:
  // Internal type used in macros to dispatch to the `operator|` overload.
  struct Helper {};

  VLoggingStream(llvm::raw_ostream* stream)
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
  [[noreturn]] auto Done() -> void;

  llvm::raw_ostream* stream_;
};

}  // namespace Carbon::Internal

// Raw logging stream. This should be used when building forms of vlog
// macros.
#define CARBON_VLOG_INTERNAL_STREAM(stream)    \
  Carbon::Internal::VLoggingStream::Helper() | \
      Carbon::Internal::VLoggingStream(stream)

#endif  // CARBON_COMMON_VLOG_INTERNAL_H_
