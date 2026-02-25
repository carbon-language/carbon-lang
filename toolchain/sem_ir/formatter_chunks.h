// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FORMATTER_CHUNKS_H_
#define CARBON_TOOLCHAIN_SEM_IR_FORMATTER_CHUNKS_H_

#include <string>

#include "common/check.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon::SemIR {

// Manages the chunks created by the formatter.
//
// There are two kinds of `Chunk`s:
// - Parent `Chunk`s, with children in a vector.
// - Content `Chunk`s, with content in a string.
//
// Initially, `AddParent` is called one or more times. Then `StartContent` is
// called once to switch modes. Once content is started, `out()` adds text to
// content `Chunk`s. `FormatChildContent` may be used to create a content
// `Chunk` which is a child of a parent.
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

  // An empty `ChunkId`.
  static constexpr ChunkId None = ChunkId(-1);

  // Reserves space for at least `count` chunks.
  auto Reserve(size_t count) -> void { chunks_.reserve(count); }

  // Adds a parent `Chunk` and returns its `ChunkId`. If `child_chunk_id` isn't
  // `None`, it's added as a child. Must be called before `StartContent`.
  auto AddParent(ChunkId child_chunk_id = None) -> ChunkId;

  // Switches from adding parents to adding content.
  auto StartContent() -> void;

  // Adds a new content `Chunk`. If `parent_chunk_id` is included in
  // output, it'll also include the new chunk. Calls `format` to support adding
  // content to the new chunk. Must be called after `StartContent`.
  auto FormatChildContent(ChunkId parent_chunk_id,
                          llvm::function_ref<auto()->void> format) -> void;

  // Adds `child_chunk_id` to the children of a `FormatChildContent`'s
  // `parent_chunk_id` if called during `format`, or otherwise includes the
  // chunk in output.
  auto AppendChildToCurrentParent(ChunkId child_chunk_id) -> void;

  // Writes included chunks to the given stream.
  auto Write(llvm::raw_ostream& stream) -> void;

  // Returns a stream to write to a content `Chunk`. The returned reference is only valid
  // until the next `Chunk` starts. Must be called after `StartContent`.
  //
  // This adds a new, parentless content `Chunk` if there's not a ready content
  // `Chunk` to write to.
  auto out() -> llvm::raw_ostream& {
    CARBON_CHECK(content_start_id_ != None);
    if (!out_) {
      AddContent(/*include_in_output=*/true);
    }
    return *out_;
  }

  auto size() -> size_t { return chunks_.size(); }

 private:
  // Either a parent or content.
  struct Chunk {
    // Whether this chunk is known to be included in the output.
    bool include_in_output;

    // Either children or content.
    std::variant<llvm::SmallVector<ChunkId>, std::string> data;
  };

  // Adds a `Chunk` that will have `content`, and directs `out_` to it.
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
