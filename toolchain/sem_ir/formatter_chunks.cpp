// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/formatter_chunks.h"

#include "common/check.h"

namespace Carbon::SemIR {

auto FormatterChunks::FlushChunk() -> void {
  CARBON_CHECK(output_chunks_.back().chunk.empty());
  output_chunks_.back().chunk = std::move(buffer_);
  buffer_.clear();
}

auto FormatterChunks::AddChunkNoFlush(bool include_in_output) -> ChunkId {
  CARBON_CHECK(buffer_.empty());
  output_chunks_.push_back({.include_in_output = include_in_output});
  return ChunkId{.index = output_chunks_.size() - 1};
}

auto FormatterChunks::AddChunk(bool include_in_output) -> ChunkId {
  FlushChunk();
  return AddChunkNoFlush(include_in_output);
}

auto FormatterChunks::AddTentativeChunkWithChild(ChunkId child_chunk)
    -> ChunkId {
  auto chunk = AddChunkNoFlush(/*include_in_output=*/false);
  output_chunks_[chunk.index].dependencies.push_back(child_chunk);
  return chunk;
}

auto FormatterChunks::FormatTentativeChunkWithParent(
    ChunkId parent_chunk, llvm::function_ref<auto()->void> format) -> void {
  CARBON_CHECK(output_chunks_.back().include_in_output,
               "All non-included chunks must be added first.");

  // If the parent is already included, we don't need to make a chunk.
  if (output_chunks_[parent_chunk.index].include_in_output) {
    format();
    return;
  }

  // Otherwise, create a new chunk and include it only if the parent is later
  // found to be used.
  auto chunk = AddChunk(false);
  output_chunks_[parent_chunk.index].dependencies.push_back(chunk);
  format();
  auto next_chunk = AddChunk(true);
  CARBON_CHECK(next_chunk.index == chunk.index + 1, "Nested FormatChildChunk");
}

auto FormatterChunks::IncludeChunkInOutput(ChunkId chunk) -> void {
  CARBON_CHECK(chunk.index != output_chunks_.size() - 1,
               "Should only be called on earlier chunks");

  if (auto& current_chunk = output_chunks_.back();
      !current_chunk.include_in_output) {
    current_chunk.dependencies.push_back(chunk);
    return;
  }

  llvm::SmallVector<ChunkId> to_add = {chunk};
  while (!to_add.empty()) {
    auto& chunk_ref = output_chunks_[to_add.pop_back_val().index];
    if (chunk_ref.include_in_output) {
      continue;
    }
    chunk_ref.include_in_output = true;
    to_add.append(chunk_ref.dependencies);
    chunk_ref.dependencies.clear();
  }
}

auto FormatterChunks::Write(llvm::raw_ostream& stream) -> void {
  FlushChunk();
  for (const auto& chunk : output_chunks_) {
    if (chunk.include_in_output) {
      stream << chunk.chunk;
    }
  }
}

}  // namespace Carbon::SemIR
