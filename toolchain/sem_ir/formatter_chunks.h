// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FORMATTER_CHUNKS_H_
#define CARBON_TOOLCHAIN_SEM_IR_FORMATTER_CHUNKS_H_

#include <string>

#include "common/check.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/base/value_store.h"

namespace Carbon::SemIR {

// Manages the chunks created by the formatter.
//
// There are two kinds of `Chunk`s:
// - Parent `Chunk`s, with children in a vector.
// - Content `Chunk`s, with content in a string.
//
// Initially, `AddParent` is called one or more times. Then `StartContent` is
// called once to switch modes, and creates an initial content `Chunk`. After
// that, either content may be written to `out()` immediately (going to the
// current content `Chunk`), or `FormatChildContent` may be used to create a
// content `Chunk` which is a child of a parent.
//
// Content `Chunk`s that are created implicitly (not through
// `FormatChildContent`) are automatically included in output. Other `Chunk`s
// are included only when marked as a child of a `Chunk` that is included, using
// `AppendChildToCurrentParent`.
class FormatterChunks {
 public:
  // A type-safe index into `chunks_`.
  struct ChunkId {
    auto operator==(const ChunkId& other) const -> bool = default;

    size_t index;
  };

  // Either a parent or content.
  struct Chunk {
    // Whether this chunk is known to be included in the output.
    bool include_in_output;

    // Either children or content.
    std::variant<llvm::SmallVector<ChunkId>, std::string> data;
  };

  // An empty `ChunkId`.
  static constexpr ChunkId None = ChunkId(-1);

  // Reserves space for at least `count` chunks.
  auto Reserve(size_t count) -> void { chunks_.reserve(count); }

  // Adds a `Chunk` that can have `children`. It can optionally start with one
  // `child_chunk`.
  auto AddParent(ChunkId child_chunk_id = None) -> ChunkId;

  // Switches from adding parents to adding content. This immediately makes
  // `out()` valid.
  auto StartContent() -> void;

  // Adds a new content `Chunk`. If the `parent_chunk` is included in
  // output, it'll also include the new chunk. Calls `format` to support adding
  // content to the new chunk.
  auto FormatChildContent(ChunkId parent_chunk_id,
                          llvm::function_ref<auto()->void> format) -> void;

  // Marks the given chunk as being included in the output if the current chunk
  // is. When `FormatChildContent` is currently active, there's a parent that
  // will determine `include_in_output for the child; otherwise, the child is
  // always included.
  //
  // For example, instructions in the file scope are added to implicitly-created
  // `Chunks` (`FormatChildContent` is not used). When
  // `AppendChildToCurrentParent` is called in that context, there's no parent
  // to limit visibility, so the child is also included.
  auto AppendChildToCurrentParent(ChunkId child_chunk_id) -> void;

  // Writes included chunks to the given stream.
  auto Write(llvm::raw_ostream& stream) -> void;

  // Returns a stream to write to the current chunk. Only valid to use after
  // `StartContent`, and may add a new chunk if one hasn't been started.
  auto out() -> llvm::raw_ostream& {
    CARBON_CHECK(content_start_id_ != None);
    if (!out_) {
      AddContent(/*include_in_output=*/true);
    }
    return *out_;
  }

  auto size() -> size_t { return chunks_.size(); }

 private:
  // Adds a `Chunk` that will have `content`.
  auto AddContent(bool include_in_output) -> ChunkId;

  // Adds `child_chunk_id` to the children of `parent_chunk_id`.
  auto AppendChildToParent(ChunkId child_chunk_id, ChunkId parent_chunk_id)
      -> void;

  // Indexes into `chunks_`.
  auto Get(ChunkId chunk_id) -> Chunk& { return chunks_[chunk_id.index]; }

  // An output stream pointing at the current content `Chunk`.
  std::unique_ptr<llvm::raw_string_ostream> out_;

  // The location where content started. Set by `StartContent`.
  ChunkId content_start_id_ = None;

  // The current parent `Chunk`. This is only set during calls to
  // `FormatChildContent`.
  ChunkId current_parent_id_ = None;

  // A sequential ordering of `Chunk`s. This will have all parent `Chunk`s
  // first, followed by content `Chunk`s at `content_start_`.
  llvm::SmallVector<Chunk> chunks_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FORMATTER_CHUNKS_H_
