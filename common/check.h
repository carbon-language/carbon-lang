// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_CHECK_H_
#define COMMON_CHECK_H_

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// Wraps a stream and exiting for fatal errors.
class ExitingStream {
 public:
  // Indicates that the program is exiting due to a bug in the program, rather
  // than, e.g., invalid input.
  ExitingStream& TreatAsBug() {
    treat_as_bug = true;
    return *this;
  }

  // If the bool cast occurs, it's because the condition is false. This supports
  // && short-circuiting the creation of ExitingStream.
  explicit operator bool() const { return true; }

  // Forward output to llvm::errs.
  template <typename T>
  ExitingStream& operator<<(const T& message) {
    if (separator) {
      llvm::errs() << ": ";
      separator = false;
    }
    llvm::errs() << message;
    return *this;
  }

  // A tag type that triggers a special overload when streaming, see below.
  struct AddSeparator {};

  // When the `AddSeparator` type is streamed, some initial input is in and this
  // is where a ": " should be added if the user streams any further
  // information. Primarily used when building macros around these streams.
  ExitingStream& operator<<(AddSeparator /*unused*/) {
    separator = true;
    return *this;
  }

  // Below are implementation details used by the macros in this header.

  // Internal type used in macros to dispatch to the `operator&` overload below.
  struct Helper {};

  // Low-precedence binary operator overload used in macros below to flush the
  // output and exit the program. We do this in a binary operator rather than
  // the destructor to ensure good debug info and backtraces for errors.
  [[noreturn]] friend auto operator|(Helper /*unused*/, ExitingStream& rhs) {
    // Finish with a newline.
    llvm::errs() << "\n";
    if (rhs.treat_as_bug) {
      std::abort();
    } else {
      std::exit(-1);
    }
  }

  [[noreturn]] ~ExitingStream() {
    llvm_unreachable(
        "Exiting streams should only be constructed with the below macros that "
        "ensure the special operator| exits the program prior to their "
        "destruction!");
  }

 private:
  // Whether a separator should be printed if << is used again.
  bool separator = false;

  // Whether the program is exiting due to a bug.
  bool treat_as_bug = false;
};

// Raw exiting stream. This should be used when building other forms of exiting
// macros like those below. It evaluates to a temporary `ExitingStream` object
// that can be manipulated, streamed into, and then will exit the program.
#define RAW_EXITING_STREAM() \
  Carbon::ExitingStream::Helper() | Carbon::ExitingStream()

// Checks the given condition, and if it's false, prints a stack, streams the
// error message, then exits. This should be used for unexpected errors, such as
// a bug in the application.
//
// For example:
//   CHECK(is_valid) << "Data is not valid!";
#define CHECK(condition)                            \
  (condition) ? (void)0                             \
              : RAW_EXITING_STREAM().TreatAsBug()   \
                    << "CHECK failure: " #condition \
                    << Carbon::ExitingStream::AddSeparator()

// This is similar to CHECK, but is unconditional. Writing FATAL() is clearer
// than CHECK(false) because it avoids confusion about control flow.
//
// For example:
//   FATAL() << "Unreachable!";
#define FATAL() RAW_EXITING_STREAM().TreatAsBug() << "FATAL: "

}  // namespace Carbon

#endif  // COMMON_CHECK_H_
