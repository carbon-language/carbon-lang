// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_NULL_DIAGNOSTICS_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_NULL_DIAGNOSTICS_H_

#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon {

template <typename LocT>
inline auto NullDiagnosticConverter() -> DiagnosticConverter<LocT>& {
  struct Converter : public DiagnosticConverter<LocT> {
    auto ConvertLoc(LocT /*loc*/,
                    DiagnosticConverter<LocT>::ContextFnT /*context_fn*/) const
        -> DiagnosticLoc override {
      return {};
    }
  };
  static auto* converter = new Converter;
  return *converter;
}

inline auto NullDiagnosticConsumer() -> DiagnosticConsumer& {
  struct Consumer : DiagnosticConsumer {
    auto HandleDiagnostic(Diagnostic /*d*/) -> void override {}
  };
  static auto* consumer = new Consumer;
  return *consumer;
}

template <typename LocT>
inline auto NullDiagnosticEmitter() -> DiagnosticEmitter<LocT>& {
  static auto* emitter = new DiagnosticEmitter<LocT>(
      NullDiagnosticConverter<LocT>(), NullDiagnosticConsumer());
  return *emitter;
}

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_NULL_DIAGNOSTICS_H_
