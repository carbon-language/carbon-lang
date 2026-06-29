// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/location.h"

#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include "toolchain/sem_ir/absolute_node_ref.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

namespace {
struct FileInfo {
  const SemIR::File* sem_ir;
  clang::SourceLocation start_loc;
};
}  // namespace

// Map a CheckIRId into information about the corresponding file in both SemIR
// and Clang's source manager.
static auto GetFileInfo(Context& context, SemIR::CheckIRId ir_id) -> FileInfo {
  const SemIR::File* sem_ir = &context.sem_ir();
  int file_index = 0;

  // If the file is imported, locate it in our imports map.
  if (ir_id != context.sem_ir().check_ir_id()) {
    auto import_id = context.check_ir_map().Get(ir_id);
    CARBON_CHECK(import_id.has_value());
    file_index = context.import_irs().GetRawIndex(import_id) + 1;

    sem_ir = context.import_irs().Get(import_id).sem_ir;
    CARBON_CHECK(sem_ir, "Node location in nonexistent IR");
  }

  // If we've seen this file before, reuse the same FileID.
  auto& file_start_locs = context.cpp_context()->carbon_file_locations();
  if (static_cast<int>(file_start_locs.size()) <= file_index) {
    // Never valid; prepare a slot for the caching below.
    file_start_locs.resize(file_index + 1);
  } else if (file_start_locs[file_index].isValid()) {
    return {.sem_ir = sem_ir, .start_loc = file_start_locs[file_index]};
  }

  // We've not seen this file before. Create a corresponding virtual file in
  // Clang's source manager.
  // TODO: Consider recreating the complete import path instead of only the
  // final entry.
  const auto& source = sem_ir->parse_tree().tokens().source();
  auto& src_mgr = context.ast_context().getSourceManager();
  auto file_id = src_mgr.createFileID(
      llvm::MemoryBufferRef(source.text(), source.filename()));
  auto file_start_loc = src_mgr.getLocForStartOfFile(file_id);
  file_start_locs[file_index] = file_start_loc;
  return {.sem_ir = sem_ir, .start_loc = file_start_loc};
}

auto GetCppLocation(Context& context, SemIR::LocId loc_id)
    -> clang::SourceLocation {
  if (!context.sem_ir().cpp_file()) {
    return clang::SourceLocation();
  }

  // Break down the `LocId` into an import path. If that ends in a C++ location,
  // we can just return that directly.
  llvm::SmallVector<SemIR::AbsoluteNodeRef> absolute_node_refs =
      SemIR::GetAbsoluteNodeRef(&context.sem_ir(), loc_id);
  const auto& final_node = absolute_node_refs.back();
  if (final_node.is_cpp()) {
    return final_node.file()->clang_source_locs().Get(
        final_node.clang_source_loc_id());
  }

  if (!final_node.node_id().has_value()) {
    // A non-existent NodeID implies our C++ is compiler-synthesised. Synthetic
    // code doesn't have a physical source location to retrieve, so the Clang
    // mapping is empty.
    return clang::SourceLocation();
  }

  // This is a location in Carbon code; get or create a corresponding file in
  // Clang and build a corresponding location.
  auto [ir, start_loc] = GetFileInfo(context, final_node.check_ir_id());
  const auto& tree = ir->parse_tree();
  auto offset =
      tree.tokens().GetByteOffset(tree.node_token(final_node.node_id()));
  return start_loc.getLocWithOffset(offset);
}

}  // namespace Carbon::Check
