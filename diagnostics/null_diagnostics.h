// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef DIAGNOSTICS_NULL_DIAGNOSTICS_H_
#define DIAGNOSTICS_NULL_DIAGNOSTICS_H_

#include "diagnostics/diagnostic_emitter.h"

namespace Carbon {

template <typename LocationT>
inline auto NullDiagnosticLocationTranslator()
    -> DiagnosticLocationTranslator<LocationT>& {
  struct Translator : DiagnosticLocationTranslator<LocationT> {
    auto GetLocation(LocationT) -> Diagnostic::Location override { return {}; }
  };
  static Translator* translator = new Translator;
  return *translator;
}

inline auto NullDiagnosticConsumer() -> DiagnosticConsumer& {
  struct Consumer : DiagnosticConsumer {
    auto HandleDiagnostic(const Diagnostic& d) -> void override {}
  };
  static auto* consumer = new Consumer;
  return *consumer;
}

template <typename LocationT>
inline auto NullDiagnosticEmitter() -> DiagnosticEmitter<LocationT>& {
  static auto* emitter = new DiagnosticEmitter<LocationT>(
      NullDiagnosticLocationTranslator<LocationT>(), NullDiagnosticConsumer());
  return *emitter;
}

}  // namespace Carbon

#endif  // DIAGNOSTICS_NULL_DIAGNOSTICS_H_
