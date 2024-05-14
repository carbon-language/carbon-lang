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

  ~SortingDiagnosticConsumer() override {
    // We choose not to automatically flush diagnostics here, because they are
    // likely to refer to data that gets destroyed before the diagnostics
    // consumer is destroyed, because the diagnostics consumer is typically
    // created before the objects that diagnostics refer into are created.
    CARBON_CHECK(diagnostics_.empty())
        << "Must flush diagnostics consumer before destroying it";
  }

  // Buffers the diagnostic.
  auto HandleDiagnostic(Diagnostic diagnostic) -> void override {
    diagnostics_.push_back(std::move(diagnostic));
  }

  // Sorts and flushes buffered diagnostics.
  void Flush() override {
    llvm::stable_sort(diagnostics_,
                      [](const Diagnostic& lhs, const Diagnostic& rhs) {
                        const auto& lhs_loc = lhs.messages[0].loc;
                        const auto& rhs_loc = rhs.messages[0].loc;
                        return std::tie(lhs_loc.filename, lhs_loc.line_number,
                                        lhs_loc.column_number) <
                               std::tie(rhs_loc.filename, rhs_loc.line_number,
                                        rhs_loc.column_number);
                      });
    for (auto& diag : diagnostics_) {
      next_consumer_->HandleDiagnostic(std::move(diag));
    }
    diagnostics_.clear();
  }

 private:
  // A Diagnostic is undesirably large for inline storage by SmallVector, so we
  // specify 0.
  llvm::SmallVector<Diagnostic, 0> diagnostics_;

  DiagnosticConsumer* next_consumer_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_SORTING_DIAGNOSTIC_CONSUMER_H_
