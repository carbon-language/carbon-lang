// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/location.h"

#include "toolchain/sem_ir/absolute_node_id.h"
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
    file_index = import_id.index + 1;

    sem_ir = context.import_irs().Get(import_id).sem_ir;
    CARBON_CHECK(sem_ir, "Node location in nonexistent IR");
  }

  // If we've seen this file before, reuse the same FileID.
  auto& file_start_locs = context.cpp_carbon_file_locations();
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

  // This is a location in Carbon code; get or create a corresponding file in
  // Clang and build a corresponding location.
  auto absolute_node_id = absolute_node_ids.back();
  auto [ir, start_loc] = GetFileInfo(context, absolute_node_id.check_ir_id());
  const auto& tree = ir->parse_tree();
  auto offset =
      tree.tokens().GetByteOffset(tree.node_token(absolute_node_id.node_id()));
  return start_loc.getLocWithOffset(offset);
}

}  // namespace Carbon::Check
