// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FORMATTER_CHUNKS_H_
#define CARBON_TOOLCHAIN_SEM_IR_FORMATTER_CHUNKS_H_

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon::SemIR {

// Manages the chunks created by the formatter.
//
// All output of the formatter is stored as `OutputChunk`s. Tentative scopes,
// such as constants, may have output for instructions made optional; these are
// stored as `include_in_output=false`, but have dependencies in case they're
// later referenced. Unreferenced tentative chunks are omitted from the output.
//
// Output of the main file scope is always included in output, and causes
// referenced tentative instructions to be included in output, including
// indirect dependencies. During this, `Formatter::FormatName` will mark
// referenced instructions for inclusion.
class FormatterChunks {
 public:
  struct ChunkId {
    size_t index;
  };

  // A chunk of the buffered output.
  struct OutputChunk {
    // Whether this chunk is known to be included in the output.
    bool include_in_output;
    // The textual contents of this chunk.
    std::string chunk = std::string();
    // Indices in `ouput_chunks_` that should be included in the output if this
    // one is.
    llvm::SmallVector<ChunkId> dependencies = {};
  };

  // Flushes the buffered output to the current chunk.
  auto FlushChunk() -> void;

  // Adds a new chunk with `include_in_output`. Does not flush existing output,
  // so should only be called if there is no buffered output.
  auto AddChunkNoFlush(bool include_in_output) -> ChunkId;

  // Flushes the current chunk and add a new chunk with `include_in_output`.
  auto AddChunk(bool include_in_output) -> ChunkId;

  // Adds a new tentative `OutputChunk`. If the new chunk is included in
  // output, it'll also include `child_chunk`.
  auto AddTentativeChunkWithChild(ChunkId child_chunk) -> ChunkId;

  // Adds a new tentative `OutputChunk`. If the `parent_chunk` is included in
  // output, it'll also include the new chunk. Calls `format` to support adding
  // content to the new chunk.
  auto FormatTentativeChunkWithParent(ChunkId parent_chunk,
                                      llvm::function_ref<auto()->void> format)
      -> void;

  // Marks the given chunk as being included in the output if the current chunk
  // is.
  auto IncludeChunkInOutput(ChunkId chunk) -> void;

  // Writes included chunks to the given stream.
  auto Write(llvm::raw_ostream& stream) -> void;

  // Returns stream representing the buffer for the current chunk.
  auto out() -> llvm::raw_ostream& { return out_; }

 private:
  friend struct TentativeOutputScope;

  // The output stream buffer.
  std::string buffer_;

  // The output stream.
  llvm::raw_string_ostream out_{buffer_};

  // Chunks of output text that we have created so far.
  llvm::SmallVector<OutputChunk> output_chunks_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FORMATTER_CHUNKS_H_
