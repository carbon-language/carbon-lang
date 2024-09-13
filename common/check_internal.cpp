// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check_internal.h"

#include "common/ostream.h"
#include "llvm/Support/Signals.h"

namespace Carbon::Internal {

// Prints the buffered message.
static auto PrintAfterStackTrace(void* str) -> void {
  llvm::errs() << reinterpret_cast<char*>(str);
}

auto CheckFailImpl(const char* kind, const char* file, int line,
                   const char* condition_str, llvm::StringRef extra_message)
    -> void {
  // Render the final check string here.
  std::string message = llvm::formatv(
      "{0} failure at {1}:{2}{3}{4}{5}{6}\n", kind, file, line,
      llvm::StringRef(condition_str).empty() ? "" : ": ", condition_str,
      extra_message.empty() ? "" : ": ", extra_message);

  // Register another signal handler to print the message. This is because we
  // want it at the bottom of output, after LLVM's builtin stack output, rather
  // than the top.
  llvm::sys::AddSignalHandler(PrintAfterStackTrace,
                              const_cast<char*>(message.c_str()));
  // It's useful to exit the program with `std::abort()` for integration with
  // debuggers and other tools. We also assume LLVM's exit handling is
  // installed, which will stack trace on `std::abort()`.
  std::abort();
}

}  // namespace Carbon::Internal
