// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_CHECK_H_
#define COMMON_CHECK_H_

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace CheckInternal {

class ExitWrapper {
 public:
  ~ExitWrapper() {
    if (exiting) {
      llvm::sys::PrintStackTrace(llvm::errs());
      if (!buffer.empty()) {
        llvm::errs() << ": " << buffer;
      }
      exit(-1);
    }
  }

  explicit operator bool() const { return true; }

  template <typename T>
  ExitWrapper& operator<<(T input) {
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
};

}  // namespace CheckInternal

#define CHECK(condition)                                     \
  (!(condition)) && CheckInternal::ExitWrapper() << "CHECK " \
                                                    "failure: " #condition

#endif  // COMMON_CHECK_H_
