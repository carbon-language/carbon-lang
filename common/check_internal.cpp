// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check_internal.h"

namespace Carbon::Internal {

// Prints the buffered message.
auto PrintAfterStackTrace(void* str) -> void {
  llvm::errs() << reinterpret_cast<char*>(str);
}

ExitingStream::~ExitingStream() {
  llvm_unreachable(
      "Exiting streams should only be constructed by check.h macros that "
      "ensure the special operator| exits the program prior to their "
      "destruction!");
}

auto ExitingStream::Done() -> void {
  buffer_ << "\n";
  // Register another signal handler to print the buffered message. This is
  // because we want it at the bottom of output, after LLVM's builtin stack
  // output, rather than the top.
  llvm::sys::AddSignalHandler(PrintAfterStackTrace,
                              const_cast<char*>(buffer_str_.c_str()));
  // It's useful to exit the program with `std::abort()` for integration with
  // debuggers and other tools. We also assume LLVM's exit handling is
  // installed, which will stack trace on `std::abort()`.
  std::abort();
}

}  // namespace Carbon::Internal
