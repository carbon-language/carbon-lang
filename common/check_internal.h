// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_CHECK_INTERNAL_H_
#define CARBON_COMMON_CHECK_INTERNAL_H_

#include <cstdlib>

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon::Internal {

// Wraps a stream and exiting for fatal errors. Should only be used by check.h
// macros.
class ExitingStream {
 public:
  // A tag type that renders as ": " in an ExitingStream, but only if it is
  // followed by additional output. Otherwise, it renders as "". Primarily used
  // when building macros around these streams.
  struct AddSeparator {};

  // Internal type used in macros to dispatch to the `operator|` overload.
  struct Helper {};

  ExitingStream()
      // Prefix the buffer with the current bug report message.
      : buffer_(buffer_str_) {}

  // Never called.
  [[noreturn]] ~ExitingStream();

  // If the bool cast occurs, it's because the condition is false. This supports
  // && short-circuiting the creation of ExitingStream.
  explicit operator bool() const { return true; }

  // Forward output to llvm::errs.
  template <typename T>
  auto operator<<(const T& message) -> ExitingStream& {
    if (separator_) {
      buffer_ << ": ";
      separator_ = false;
    }
    buffer_ << message;
    return *this;
  }

  auto operator<<(AddSeparator /*add_separator*/) -> ExitingStream& {
    separator_ = true;
    return *this;
  }

  // Low-precedence binary operator overload used in check.h macros to flush the
  // output and exit the program. We do this in a binary operator rather than
  // the destructor to ensure good debug info and backtraces for errors.
  [[noreturn]] friend auto operator|(Helper /*helper*/, ExitingStream& stream)
      -> void {
    stream.Done();
  }

 private:
  [[noreturn]] auto Done() -> void;

  // Whether a separator should be printed if << is used again.
  bool separator_ = false;

  std::string buffer_str_;
  llvm::raw_string_ostream buffer_;
};

}  // namespace Carbon::Internal

// Raw exiting stream. This should be used when building forms of exiting
// macros. It evaluates to a temporary `ExitingStream` object that can be
// manipulated, streamed into, and then will exit the program.
#define CARBON_CHECK_INTERNAL_STREAM() \
  Carbon::Internal::ExitingStream::Helper() | Carbon::Internal::ExitingStream()

#endif  // CARBON_COMMON_CHECK_INTERNAL_H_
