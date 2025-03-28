// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_NULL_DIAGNOSTICS_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_NULL_DIAGNOSTICS_H_

#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::Diagnostics {

// Returns a singleton consumer that doesn't print its diagnostics.
inline auto NullConsumer() -> Consumer& {
  struct SingletonConsumer : Consumer {
    auto HandleDiagnostic(Diagnostic /*d*/) -> void override {}
  };
  static auto* consumer = new SingletonConsumer;
  return *consumer;
}

// Returns a singleton emitter that doesn't print its diagnostics.
template <typename LocT>
inline auto NullEmitter() -> Emitter<LocT>& {
  class SingletonEmitter : public Emitter<LocT> {
   public:
    using Emitter<LocT>::Emitter;

   protected:
    // Converts a filename directly to the diagnostic location.
    auto ConvertLoc(LocT /*loc*/,
                    Emitter<LocT>::ContextFnT /*context_fn*/) const
        -> ConvertedLoc override {
      return {.loc = {}, .last_byte_offset = -1};
    }
  };

  static auto* emitter = new SingletonEmitter(&NullConsumer());
  return *emitter;
}

}  // namespace Carbon::Diagnostics

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_NULL_DIAGNOSTICS_H_
