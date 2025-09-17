// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/location.h"

#include "toolchain/sem_ir/absolute_node_id.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

static auto GetFile(Context& context, SemIR::CheckIRId ir_id)
    -> const SemIR::File* {
  if (ir_id == context.sem_ir().check_ir_id()) {
    // Common case: the IR is the current file.
    return &context.sem_ir();
  }

  // If the file is imported, locate it in our imports map.
  auto import_id = context.check_ir_map().Get(ir_id);
  if (!import_id.has_value()) {
    // We never imported this CheckIR.
    // TODO: Can this happen?
    return nullptr;
  }
  return context.import_irs().Get(import_id).sem_ir;
}

auto GetCppLocation(Context& context, SemIR::LocId loc_id)
    -> clang::SourceLocation {
  if (!context.sem_ir().clang_ast_unit()) {
    return clang::SourceLocation();
  }

  // Break down the `LocId` into an import path. If that ends in a C++ location,
  // we can just return that directly.
  llvm::SmallVector<SemIR::AbsoluteNodeId> absolute_node_ids =
      SemIR::GetAbsoluteNodeId(&context.sem_ir(), loc_id);
  if (absolute_node_ids.back().check_ir_id() == SemIR::CheckIRId::Cpp) {
    return context.sem_ir().clang_source_locs().Get(
        absolute_node_ids.back().clang_source_loc_id());
  }

  // This is a location in Carbon code; decompose it so we can map it into a
  // Clang location.
  // TODO: Consider recreating the complete import path instead of only the
  // final entry.
  auto absolute_node_id = absolute_node_ids.back();
  const auto* ir = GetFile(context, absolute_node_id.check_ir_id());
  if (!ir) {
    return clang::SourceLocation();
  }
  const auto& tree = ir->parse_tree();
  const auto& source = tree.tokens().source();
  auto offset =
      tree.tokens().GetByteOffset(tree.node_token(absolute_node_id.node_id()));

  // Get or create a corresponding Clang file.
  // TODO: Consider caching a mapping from Carbon ImportIRIds to Clang
  // start-of-file SourceLocations.
  auto& src_mgr = context.ast_context().getSourceManager();
  auto file = src_mgr.getFileManager().getOptionalFileRef(source.filename());
  if (!file) {
    file = src_mgr.getFileManager().getVirtualFileRef(
        source.filename(), static_cast<off_t>(0), static_cast<time_t>(0));
  }
  src_mgr.overrideFileContents(
      *file, llvm::MemoryBufferRef(source.text(), source.filename()));

  // Build a corresponding location.
  auto file_id = src_mgr.getOrCreateFileID(
      *file, clang::SrcMgr::CharacteristicKind::C_User);
  return src_mgr.getLocForStartOfFile(file_id).getLocWithOffset(offset);
}

}  // namespace Carbon::Check
