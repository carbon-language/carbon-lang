// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_NULL_DIAGNOSTICS_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_NULL_DIAGNOSTICS_H_

#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon {

// Returns a singleton consumer that doesn't print its diagnostics.
inline auto NullDiagnosticConsumer() -> DiagnosticConsumer& {
  struct Consumer : DiagnosticConsumer {
    auto HandleDiagnostic(Diagnostic /*d*/) -> void override {}
  };
  static auto* consumer = new Consumer;
  return *consumer;
}

// Returns a singleton emitter that doesn't print its diagnostics.
template <typename LocT>
inline auto NullDiagnosticEmitter() -> DiagnosticEmitter<LocT>& {
  class Emitter : public DiagnosticEmitter<LocT> {
   public:
    using DiagnosticEmitter<LocT>::DiagnosticEmitter;

   protected:
    // Converts a filename directly to the diagnostic location.
    auto ConvertLoc(LocT /*loc*/,
                    DiagnosticEmitter<LocT>::ContextFnT /*context_fn*/) const
        -> ConvertedDiagnosticLoc override {
      return {.loc = {}, .last_byte_offset = -1};
    }
  };

  static auto* emitter = new Emitter(&NullDiagnosticConsumer());
  return *emitter;
}

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_NULL_DIAGNOSTICS_H_
