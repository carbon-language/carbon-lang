// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_SEM_IR_DIAGNOSTIC_CONVERTER_H_
#define CARBON_TOOLCHAIN_CHECK_SEM_IR_DIAGNOSTIC_CONVERTER_H_

#include "llvm/ADT/ArrayRef.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/diagnostics/diagnostic_converter.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

// Handles the transformation of a SemIRLoc to a DiagnosticLoc.
class SemIRDiagnosticConverter : public DiagnosticConverter<SemIRLoc> {
 public:
  using TreeFnT = llvm::function_ref<const Parse::TreeAndSubtrees&()>;

  explicit SemIRDiagnosticConverter(
      llvm::ArrayRef<TreeFnT> imported_trees_and_subtrees,
      const SemIR::File* sem_ir)
      : imported_trees_and_subtrees_(imported_trees_and_subtrees),
        sem_ir_(sem_ir) {}

  // Implements `DiagnosticConverter::ConvertLoc`. Adds context for any imports
  // used in the current SemIR to get to the underlying code.
  //
  // For the last byte offset, this uses `last_token_` exclusively for imported
  // locations, or `loc` if it's in the same file and (for whatever reason)
  // later.
  auto ConvertLoc(SemIRLoc loc, ContextFnT context_fn) const
      -> ConvertedDiagnosticLoc override;

  // Implements argument conversions for supported check-phase arguments.
  auto ConvertArg(llvm::Any arg) const -> llvm::Any override;

  // If a byte offset is past the current last byte offset, advances forward.
  // Earlier offsets are ignored.
  auto AdvanceToken(Lex::TokenIndex token) -> void {
    last_token_ = std::max(last_token_, token);
  }

 private:
  // Implements `ConvertLoc`, but without `last_token_` applied.
  auto ConvertLocImpl(SemIRLoc loc, ContextFnT context_fn) const
      -> ConvertedDiagnosticLoc;

  // Converts a node_id corresponding to a specific sem_ir to a diagnostic
  // location.
  auto ConvertLocInFile(const SemIR::File* sem_ir, Parse::NodeId node_id,
                        bool token_only, ContextFnT context_fn) const
      -> ConvertedDiagnosticLoc;

  // Converters for each SemIR.
  llvm::ArrayRef<TreeFnT> imported_trees_and_subtrees_;

  // The current SemIR being processed.
  const SemIR::File* sem_ir_;

  // The last token encountered during processing.
  Lex::TokenIndex last_token_ = Lex::TokenIndex::None;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_SEM_IR_DIAGNOSTIC_CONVERTER_H_
