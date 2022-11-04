// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_CHECK_INTERNAL_H_
#define CARBON_COMMON_CHECK_INTERNAL_H_

#include <cstdlib>

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/PrettyStackTrace.h"
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
      : buffer_str_(llvm::getBugReportMsg()), buffer_(buffer_str_) {}

  [[noreturn]] ~ExitingStream() {
    llvm_unreachable(
        "Exiting streams should only be constructed by check.h macros that "
        "ensure the special operator| exits the program prior to their "
        "destruction!");
  }

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
  [[noreturn]] friend auto operator|(Helper /*helper*/, ExitingStream& stream) {
    stream.buffer_ << "\n";
    // The buffer will start with the bug report message; we replace that so
    // that the streamed message is printed _between_ the default bug report
    // message and stack information.
    llvm::setBugReportMsg(stream.buffer_str_.c_str());
    // It's useful to exit the program with `std::abort()` for integration with
    // debuggers and other tools. We also assume LLVM's exit handling is
    // installed, which will stack trace on `std::abort()`.
    std::abort();
  }

 private:
  // Whether a separator should be printed if << is used again.
  bool separator_ = false;

  std::string buffer_str_;
  llvm::raw_string_ostream buffer_;
};

}  // namespace Carbon::Internal

#endif  // CARBON_COMMON_CHECK_INTERNAL_H_
