// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_DIAGNOSTIC_LISTENER_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_DIAGNOSTIC_LISTENER_H_

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Check {

class CarbonClangDiagnosticConsumer;

// Interface for listening to Clang diagnostics from
// CarbonClangDiagnosticConsumer.
//
// Pushes itself onto the consumer's listener stack on construction and pops
// itself on destruction.
class CppDiagnosticListener {
 public:
  explicit CppDiagnosticListener(CarbonClangDiagnosticConsumer& consumer);
  virtual ~CppDiagnosticListener();

  // Handles a Clang diagnostic and any associated code snippet.
  virtual auto HandleDiagnostic(clang::DiagnosticsEngine::Level diag_level,
                                const clang::Diagnostic& info,
                                llvm::StringRef message,
                                llvm::StringRef snippet) -> void = 0;

  // Flushes any buffered diagnostics.
  virtual auto Flush() -> void {}

 protected:
  auto consumer() -> CarbonClangDiagnosticConsumer& { return *consumer_; }

 private:
  CarbonClangDiagnosticConsumer* consumer_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_DIAGNOSTIC_LISTENER_H_
