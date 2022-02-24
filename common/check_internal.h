// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_CHECK_INTERNAL_H_
#define COMMON_CHECK_INTERNAL_H_

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

  [[noreturn]] ~ExitingStream() {
    llvm_unreachable(
        "Exiting streams should only be constructed by check.h macros that "
        "ensure the special operator| exits the program prior to their "
        "destruction!");
  }

  // Indicates that the program is exiting due to a bug in the program, rather
  // than, e.g., invalid input.
  auto TreatAsBug() -> ExitingStream& {
    treat_as_bug_ = true;
    return *this;
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

  auto operator<<(AddSeparator /*unused*/) -> ExitingStream& {
    separator_ = true;
    return *this;
  }

  // Low-precedence binary operator overload used in check.h macros to flush the
  // output and exit the program. We do this in a binary operator rather than
  // the destructor to ensure good debug info and backtraces for errors.
  [[noreturn]] friend auto operator|(Helper /*unused*/, ExitingStream& rhs) {
    // Finish with a newline.
    llvm::errs() << "\n";
    if (rhs.treat_as_bug_) {
      std::abort();
    } else {
      std::exit(-1);
    }
  }

 private:
  // Whether a separator should be printed if << is used again.
  bool separator_ = false;

  // Whether the program is exiting due to a bug.
  bool treat_as_bug_ = false;
};

}  // namespace Carbon::Internal

#endif  // COMMON_CHECK_INTERNAL_H_
