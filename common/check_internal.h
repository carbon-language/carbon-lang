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

  ExitingStream() {
    // Start all messages with a stack trace. Printing this first ensures the
    // error message is the last thing written which makes it easier to find. It
    // also helps ensure we report the most useful stack trace and location
    // information.
    llvm::errs() << "Stack trace:\n";
    llvm::sys::PrintStackTrace(llvm::errs());
  }

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
      llvm::errs() << ": ";
      separator_ = false;
    }
    llvm::errs() << message;
    return *this;
  }

  auto operator<<(AddSeparator /*add_separator*/) -> ExitingStream& {
    separator_ = true;
    return *this;
  }

  // Low-precedence binary operator overload used in check.h macros to flush the
  // output and exit the program. We do this in a binary operator rather than
  // the destructor to ensure good debug info and backtraces for errors.
  [[noreturn]] friend auto operator|(Helper /*helper*/,
                                     ExitingStream& /*rhs*/) {
    // Finish with a newline.
    llvm::errs() << "\n";
    // We assume LLVM's exit handling is installed, which will stack trace on
    // `std::abort()`. We print a more user friendly stack trace on
    // construction, but it is still useful to exit the program with
    // `std::abort()` for integration with debuggers and other tools. We also
    // want to do any pending cleanups. So we replicate the signal handling here
    // and unregister LLVM's handlers right before we abort.
    llvm::sys::RunInterruptHandlers();
    llvm::sys::unregisterHandlers();
    std::abort();
  }

 private:
  // Whether a separator should be printed if << is used again.
  bool separator_ = false;
};

}  // namespace Carbon::Internal

#endif  // CARBON_COMMON_CHECK_INTERNAL_H_
