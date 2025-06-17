// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DIAGNOSTIC_EMITTER_H_
#define CARBON_TOOLCHAIN_CHECK_DIAGNOSTIC_EMITTER_H_

#include "llvm/ADT/ArrayRef.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/sem_ir/diagnostic_loc_converter.h"

namespace Carbon::Check {

// Handles the transformation of a SemIR::LocId to a DiagnosticLoc.
class DiagnosticEmitter : public DiagnosticEmitterBase {
 public:
  explicit DiagnosticEmitter(
      Diagnostics::Consumer* consumer,
      llvm::ArrayRef<Parse::GetTreeAndSubtreesFn> tree_and_subtrees_getters,
      const SemIR::File* sem_ir)
      : DiagnosticEmitterBase(consumer),
        sem_ir_(sem_ir),
        loc_converter_(tree_and_subtrees_getters, sem_ir) {}

  // If a byte offset is past the current last byte offset, advances forward.
  // Earlier offsets are ignored.
  auto AdvanceToken(Lex::TokenIndex token) -> void {
    last_token_ = std::max(last_token_, token);
  }

 protected:
  // Implements argument conversions for supported check-phase arguments.
  auto ConvertArg(llvm::Any arg) const -> llvm::Any override;

  // Implements `DiagnosticConverter::ConvertLoc`. Adds context for any imports
  // used in the current SemIR to get to the underlying code.
  //
  // For the last byte offset, this uses `last_token_` exclusively for imported
  // locations, or `loc` if it's in the same file and (for whatever reason)
  // later.
  auto ConvertLoc(LocIdForDiagnostics loc_id, ContextFnT context_fn) const
      -> Diagnostics::ConvertedLoc override;

 private:
  // The current SemIR being processed.
  const SemIR::File* sem_ir_;

  // Converter for locations.
  SemIR::DiagnosticLocConverter loc_converter_;

  // The last token encountered during processing.
  Lex::TokenIndex last_token_ = Lex::TokenIndex::None;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DIAGNOSTIC_EMITTER_H_
