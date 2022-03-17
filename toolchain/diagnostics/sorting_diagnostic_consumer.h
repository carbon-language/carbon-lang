// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_DIAGNOSTICS_SORTING_DIAGNOSTIC_CONSUMER_H_
#define TOOLCHAIN_DIAGNOSTICS_SORTING_DIAGNOSTIC_CONSUMER_H_

#include "llvm/ADT/STLExtras.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon {

// Buffers incoming diagnostics for printing and sorting.
class SortingDiagnosticConsumer : public DiagnosticConsumer {
 public:
  explicit SortingDiagnosticConsumer(DiagnosticConsumer& next_consumer)
      : next_consumer_(&next_consumer) {}

  auto HandleDiagnostic(const Diagnostic& diagnostic) -> void override {
    diagnostics_.push_back(diagnostic);
  }

  // Sorts and flushes buffered diagnostics.
  auto SortAndFlush() {
    llvm::sort(diagnostics_, [](const Diagnostic& lhs, const Diagnostic& rhs) {
      if (lhs.location.line_number < rhs.location.line_number) {
        return true;
      } else if (lhs.location.line_number == rhs.location.line_number) {
        return lhs.location.column_number < rhs.location.column_number;
      } else {
        return false;
      }
    });
    for (const auto& diagnostic : diagnostics_) {
      next_consumer_->HandleDiagnostic(diagnostic);
    }
    diagnostics_.clear();
  }

 private:
  llvm::SmallVector<Diagnostic, 0> diagnostics_;
  DiagnosticConsumer* next_consumer_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_DIAGNOSTICS_SORTING_DIAGNOSTIC_CONSUMER_H_
