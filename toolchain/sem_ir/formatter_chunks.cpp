// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/formatter_chunks.h"

#include "common/check.h"

namespace Carbon::SemIR {

auto FormatterChunks::StartContent() -> void {
  CARBON_CHECK(content_start_id_ == None);
  content_start_id_ = ChunkId{.index = chunks_.size()};
}

auto FormatterChunks::AddContent(bool include_in_output) -> ChunkId {
  CARBON_CHECK(content_start_id_ != None, "Must call StartContent first");
  auto chunk_id = ChunkId{.index = chunks_.size()};
  chunks_.push_back(
      {.include_in_output = include_in_output, .data = std::string()});
  out_ = std::make_unique<llvm::raw_string_ostream>(
      std::get<std::string>(Get(chunk_id).data));
  return chunk_id;
}

auto FormatterChunks::AddParent(ChunkId child_chunk_id) -> ChunkId {
  CARBON_CHECK(content_start_id_ == None, "Already called StartContent");
  llvm::SmallVector<ChunkId> children;
  if (child_chunk_id != None) {
    children.push_back(child_chunk_id);
  }
  auto chunk_id = ChunkId{.index = chunks_.size()};
  chunks_.push_back({.include_in_output = false, .data = std::move(children)});
  return chunk_id;
}

auto FormatterChunks::FormatChildContent(
    ChunkId parent_chunk_id, llvm::function_ref<auto()->void> format) -> void {
  CARBON_CHECK(content_start_id_ != None, "Must call StartContent first");
  CARBON_CHECK(current_parent_id_ == None, "Cannot nest FormatChildContent");

  // We only need to call `AddContent` for non-included content because `out()`
  // is included by default.
  if (Get(parent_chunk_id).include_in_output) {
    format();
    return;
  }

  // Otherwise, create a content `Chunk` and include it only if the parent is
  // later found to be used.
  AppendChildToParent(AddContent(/*include_in_output=*/false), parent_chunk_id);

  current_parent_id_ = parent_chunk_id;
  format();
  current_parent_id_ = None;

  // Reset the output stream so that the next call to `out()` creates a new
  // chunk.
  out_.reset();
}

auto FormatterChunks::AppendChildToParent(ChunkId child_chunk_id,
                                          ChunkId parent_chunk_id) -> void {
  CARBON_CHECK(!Get(parent_chunk_id).include_in_output);
  auto* children =
      std::get_if<llvm::SmallVector<ChunkId>>(&Get(parent_chunk_id).data);
  CARBON_CHECK(children);
  children->push_back(child_chunk_id);
}

auto FormatterChunks::AppendChildToCurrentParent(ChunkId child_chunk_id)
    -> void {
  if (current_parent_id_ != None) {
    // If the parent is not included, add the `chunk` to the parent's children
    // for conditional inclusion.
    if (!Get(current_parent_id_).include_in_output) {
      AppendChildToParent(child_chunk_id, current_parent_id_);
      return;
    }
  }

  // If the parent is already included, or there is no parent (this is not
  // currently a tentative chunk), include the chunk and all of its children.
  llvm::SmallVector<ChunkId> to_include = {child_chunk_id};
  while (!to_include.empty()) {
    auto& chunk_ref = Get(to_include.pop_back_val());
    if (chunk_ref.include_in_output) {
      continue;
    }
    chunk_ref.include_in_output = true;
    if (auto* children =
            std::get_if<llvm::SmallVector<ChunkId>>(&chunk_ref.data)) {
      to_include.append(*children);
      children->clear();
    }
  }
}

auto FormatterChunks::Write(llvm::raw_ostream& stream) -> void {
  CARBON_CHECK(content_start_id_ != None);
  for (const auto& chunk :
       llvm::ArrayRef(chunks_).drop_front(content_start_id_.index)) {
    if (chunk.include_in_output) {
      stream << std::get<std::string>(chunk.data);
    }
  }
}

}  // namespace Carbon::SemIR
