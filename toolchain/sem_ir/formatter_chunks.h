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

  // All formatted output within the scope of this object is redirected to a
  // new tentative `OutputChunk`. The new chunk will depend on `parent_chunk`.
  struct TentativeScope {
    explicit TentativeScope(FormatterChunks* chunks, ChunkId parent_chunk);
    ~TentativeScope();

    FormatterChunks* chunks;
    ChunkId chunk;
  };

  // Flushes the buffered output to the current chunk.
  auto FlushChunk() -> void;

  // Adds a new chunk to the output. Does not flush existing output, so should
  // only be called if there is no buffered output.
  auto AddChunkNoFlush(bool include_in_output) -> ChunkId;

  // Flushes the current chunk and add a new chunk to the output.
  auto AddChunk(bool include_in_output) -> ChunkId;

  // Marks the given chunk as being included in the output if the current chunk
  // is.
  auto IncludeChunkInOutput(ChunkId chunk) -> void;

  // Write buffered output to the given stream.
  auto Write(llvm::raw_ostream& stream) -> void;

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
