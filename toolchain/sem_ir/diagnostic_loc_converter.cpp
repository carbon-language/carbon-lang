// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/diagnostic_loc_converter.h"

namespace Carbon::SemIR {

auto DiagnosticLocConverter::ConvertWithImports(LocId loc_id,
                                                bool token_only) const
    -> LocAndImports {
  llvm::SmallVector<SemIR::AbsoluteNodeId> absolute_node_ids =
      SemIR::GetAbsoluteNodeId(sem_ir_, loc_id);
  auto final_node_id = absolute_node_ids.pop_back_val();

  // Convert the final location.
  LocAndImports result = {.loc = ConvertImpl(final_node_id, token_only)};

  // Convert the import locations.
  for (const auto& absolute_node_id : absolute_node_ids) {
    if (!absolute_node_id.node_id().has_value()) {
      // TODO: Add an `ImportLoc` pointing at the prelude for the case where
      // we don't have a location.
      continue;
    }
    result.imports.push_back({.loc = ConvertImpl(absolute_node_id, false).loc});
  }

  return result;
}

auto DiagnosticLocConverter::Convert(LocId loc_id, bool token_only) const
    -> Diagnostics::ConvertedLoc {
  llvm::SmallVector<SemIR::AbsoluteNodeId> absolute_node_ids =
      SemIR::GetAbsoluteNodeId(sem_ir_, loc_id);
  return ConvertImpl(absolute_node_ids.back(), token_only);
}

auto DiagnosticLocConverter::ConvertImpl(SemIR::AbsoluteNodeId absolute_node_id,
                                         bool token_only) const
    -> Diagnostics::ConvertedLoc {
  if (absolute_node_id.check_ir_id() == SemIR::CheckIRId::Cpp) {
    return ConvertImpl(absolute_node_id.clang_source_loc_id());
  }

  return ConvertImpl(absolute_node_id.check_ir_id(), absolute_node_id.node_id(),
                     token_only);
}

auto DiagnosticLocConverter::ConvertImpl(SemIR::CheckIRId check_ir_id,
                                         Parse::NodeId node_id,
                                         bool token_only) const
    -> Diagnostics::ConvertedLoc {
  CARBON_CHECK(check_ir_id != SemIR::CheckIRId::Cpp);
  const auto& tree_and_subtrees =
      tree_and_subtrees_getters_[check_ir_id.index]();
  return tree_and_subtrees.NodeToDiagnosticLoc(node_id, token_only);
}

auto DiagnosticLocConverter::ConvertImpl(
    ClangSourceLocId clang_source_loc_id) const -> Diagnostics::ConvertedLoc {
  clang::SourceLocation clang_loc =
      sem_ir_->clang_source_locs().Get(clang_source_loc_id);

  CARBON_CHECK(sem_ir_->cpp_ast());
  clang::PresumedLoc presumed_loc =
      sem_ir_->cpp_ast()->getSourceManager().getPresumedLoc(clang_loc);
  if (presumed_loc.isInvalid()) {
    return Diagnostics::ConvertedLoc();
  }
  unsigned offset =
      sem_ir_->cpp_ast()->getSourceManager().getDecomposedLoc(clang_loc).second;

  return Diagnostics::ConvertedLoc{
      .loc = {.filename = presumed_loc.getFilename(),
              .line_number = static_cast<int32_t>(presumed_loc.getLine())},
      .last_byte_offset = static_cast<int32_t>(offset)};
}

}  // namespace Carbon::SemIR
