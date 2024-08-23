// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_SEM_IR_DIAGNOSTIC_CONVERTER_H_
#define CARBON_TOOLCHAIN_CHECK_SEM_IR_DIAGNOSTIC_CONVERTER_H_

#include "llvm/ADT/ArrayRef.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/diagnostics/diagnostic_converter.h"
#include "toolchain/parse/tree_node_diagnostic_converter.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

// Handles the transformation of a SemIRLoc to a DiagnosticLoc.
class SemIRDiagnosticConverter : public DiagnosticConverter<SemIRLoc> {
 public:
  explicit SemIRDiagnosticConverter(
      llvm::ArrayRef<Parse::NodeLocConverter> node_converters,
      const SemIR::File* sem_ir)
      : node_converters_(node_converters), sem_ir_(sem_ir) {}

  // Converts an instruction's location to a diagnostic location, which will be
  // the underlying line of code. Adds context for any imports used in the
  // current SemIR to get to the underlying code.
  auto ConvertLoc(SemIRLoc loc, ContextFnT context_fn) const
      -> DiagnosticLoc override;

  // Implements argument conversions for supported check-phase arguments.
  auto ConvertArg(llvm::Any arg) const -> llvm::Any override;

 private:
  // Converts a node_id corresponding to a specific sem_ir to a diagnostic
  // location.
  auto ConvertLocInFile(const SemIR::File* sem_ir, Parse::NodeId node_id,
                        bool token_only, ContextFnT context_fn) const
      -> DiagnosticLoc;

  // Converters for each SemIR.
  llvm::ArrayRef<Parse::NodeLocConverter> node_converters_;

  // The current SemIR being processed.
  const SemIR::File* sem_ir_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_SEM_IR_DIAGNOSTIC_CONVERTER_H_
