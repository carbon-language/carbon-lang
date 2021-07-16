// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_CHECK_H_
#define COMMON_CHECK_H_

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace CheckInternal {

// Wraps a stream and exiting for CHECK.
class ExitWrapper {
 public:
  ~ExitWrapper() {
    // If exiting will occur, print the buffer and errors.
    if (exiting) {
      llvm::sys::PrintStackTrace(llvm::errs());
      llvm::errs() << buffer << "\n";
      exit(-1);
    }
  }

  // Indicates that initial input is in, so this is where a ": " should be added
  // before user input.
  ExitWrapper& add_separator() {
    separator = true;
    return *this;
  }

  explicit operator bool() const { return true; }

  template <typename T>
  ExitWrapper& operator<<(T input) {
    if (separator) {
      buffer_stream << ": ";
      separator = false;
    }
    buffer_stream << input;
    return *this;
  }

  friend ExitWrapper& operator&&(bool cond,
                                 CheckInternal::ExitWrapper& exit_wrapper) {
    if (cond) {
      exit_wrapper.exiting = true;
    }
    return exit_wrapper;
  }

  std::string buffer;
  llvm::raw_string_ostream buffer_stream = llvm::raw_string_ostream(buffer);
  bool exiting = false;
  bool separator = false;
};

}  // namespace CheckInternal

#define CHECK(condition)                                             \
  (!(condition)) &&                                                  \
      (CheckInternal::ExitWrapper() << "CHECK failure: " #condition) \
          .add_separator()

#endif  // COMMON_CHECK_H_
