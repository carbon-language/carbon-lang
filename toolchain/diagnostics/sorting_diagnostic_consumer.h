// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_SORTING_DIAGNOSTIC_CONSUMER_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_SORTING_DIAGNOSTIC_CONSUMER_H_

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::Diagnostics {

// Buffers incoming diagnostics for printing and sorting.
//
// Sorting is based on `last_byte_offset` without taking the filename into
// account. When processing multiple files, it's expected that separate
// consumers will be used in order to keep diagnostics distinct. Typically
// `Diagnostic::messages[0]` will always be a location in the consumer's primary
// file, but if it needs to correspond to a different file, the
// `last_byte_offset` must still indicate an offset within the primary file.
class SortingConsumer : public Consumer {
 public:
  explicit SortingConsumer(Consumer& next_consumer)
      : next_consumer_(&next_consumer) {}

  ~SortingConsumer() override {
    // We choose not to automatically flush diagnostics here, because they are
    // likely to refer to data that gets destroyed before the diagnostics
    // consumer is destroyed, because the diagnostics consumer is typically
    // created before the objects that diagnostics refer into are created.
    CARBON_CHECK(diagnostics_.empty(),
                 "Must flush diagnostics consumer before destroying it");
  }

  // Buffers the diagnostic.
  auto HandleDiagnostic(Diagnostic diagnostic) -> void override {
    diagnostics_.push_back(std::move(diagnostic));
  }

  // Sorts and flushes buffered diagnostics.
  auto Flush() -> void override {
    llvm::stable_sort(diagnostics_,
                      [](const Diagnostic& lhs, const Diagnostic& rhs) {
                        return lhs.last_byte_offset < rhs.last_byte_offset;
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

  Consumer* next_consumer_;
};

}  // namespace Carbon::Diagnostics

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_SORTING_DIAGNOSTIC_CONSUMER_H_
