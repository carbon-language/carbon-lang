// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/formatter_chunks.h"

#include "common/check.h"

namespace Carbon::SemIR {

FormatterChunks::TentativeScope::TentativeScope(FormatterChunks* chunks,
                                                ChunkId parent_chunk)
    : chunks(chunks) {
  // If our parent is not known to be included, create a new chunk and
  // include it only if the parent is later found to be used.
  if (!chunks->output_chunks_[parent_chunk.index].include_in_output) {
    chunk = chunks->AddChunk(false);
    chunks->output_chunks_[parent_chunk.index].dependencies.push_back(chunk);
  }
}

FormatterChunks::TentativeScope::~TentativeScope() {
  auto next_chunk = chunks->AddChunk(true);
  CARBON_CHECK(next_chunk.index == chunk.index + 1,
               "Nested FormatterChunks::TentativeScope");
}

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
