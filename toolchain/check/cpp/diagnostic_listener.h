// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_DIAGNOSTIC_LISTENER_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_DIAGNOSTIC_LISTENER_H_

#include <string>

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"

namespace Carbon::Check {

class CarbonClangDiagnosticConsumer;

// Interface for listening to Clang diagnostics from
// CarbonClangDiagnosticConsumer.
//
// Pushes itself onto the consumer's listener stack on construction and pops
// itself on destruction.
class CppDiagnosticListener {
 public:
  struct Diagnostic {
    clang::DiagnosticsEngine::Level level;
    clang::SourceLocation location;
    std::string message;
    std::string snippet;
  };

  explicit CppDiagnosticListener(CarbonClangDiagnosticConsumer& consumer);
  virtual ~CppDiagnosticListener();

  // Emits a group of buffered diagnostics, which will comprise a single leading
  // diagnostic -- typically an error / warning / remark -- followed by a
  // sequence of attached notes. The first diagnostic can also be a note if the
  // diagnostics engine is flushed in the middle of Clang emitting a diagnostic,
  // or if Clang emits a "stray" note not attached to any diagnostic. Both would
  // generally indicate Clang bugs.
  virtual auto EmitDiagnostics(llvm::ArrayRef<Diagnostic> diags) -> void = 0;

 protected:
  auto consumer() -> CarbonClangDiagnosticConsumer& { return *consumer_; }

 private:
  CarbonClangDiagnosticConsumer* consumer_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_DIAGNOSTIC_LISTENER_H_
