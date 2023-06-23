// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_SORTING_DIAGNOSTIC_CONSUMER_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_SORTING_DIAGNOSTIC_CONSUMER_H_

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon {

// Buffers incoming diagnostics for printing and sorting.
class SortingDiagnosticConsumer : public DiagnosticConsumer {
 public:
  explicit SortingDiagnosticConsumer(DiagnosticConsumer& next_consumer)
      : next_consumer_(&next_consumer) {}

  ~SortingDiagnosticConsumer() override { Flush(); }

  // Buffers the diagnostic.
  auto HandleDiagnostic(Diagnostic diagnostic) -> void override {
    diagnostics_.push_back(std::move(diagnostic));
  }

  // Sorts and flushes buffered diagnostics.
  void Flush() override {
    llvm::stable_sort(diagnostics_,
                      [](const Diagnostic& lhs, const Diagnostic& rhs) {
                        return std::tie(lhs.message.location.line_number,
                                        lhs.message.location.column_number) <
                               std::tie(rhs.message.location.line_number,
                                        rhs.message.location.column_number);
                      });
    for (auto& diag : diagnostics_) {
      next_consumer_->HandleDiagnostic(std::move(diag));
    }
    diagnostics_.clear();
  }

 private:
  llvm::SmallVector<Diagnostic, 0> diagnostics_;
  DiagnosticConsumer* next_consumer_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_SORTING_DIAGNOSTIC_CONSUMER_H_
