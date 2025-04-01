// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_COPY_ON_WRITE_BLOCK_H_
#define CARBON_TOOLCHAIN_SEM_IR_COPY_ON_WRITE_BLOCK_H_

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// A handle to a new block that may be modified, with copy-on-write semantics.
//
// The constructor is given the ID of an existing block that provides the
// initial contents of the new block. The new block is lazily allocated; if no
// modifications have been made, the `id()` function will return the original
// block ID.
//
// This is intended to avoid an unnecessary block allocation in the case where
// the new block ends up being exactly the same as the original block.
template <typename BlockIdType, auto (SemIR::File::*ValueStore)()>
class CopyOnWriteBlock {
 public:
  struct UninitializedBlock {
    size_t size;
  };

  // Constructs the block. `source_id` is used as the initial value of the
  // block. `file` must not be null.
  explicit CopyOnWriteBlock(SemIR::File* file, BlockIdType source_id)
      : file_(file), source_id_(source_id) {}

  // Constructs the block, treating the original block as an uninitialized block
  // with `size` elements. `file` must not be null.
  explicit CopyOnWriteBlock(SemIR::File* file, UninitializedBlock uninit)
      : file_(file),
        source_id_(BlockIdType::None),
        id_((file_->*ValueStore)().AddUninitialized(uninit.size)) {}

  // Gets a block ID containing the resulting elements. Note that further
  // modifications may or may not allocate a new ID, so this should only be
  // called once all modifications have been performed.
  auto id() const -> BlockIdType { return id_; }

  // Gets a canonical block ID containing the resulting elements. This assumes
  // the original block ID, if specified, was also canonical.
  auto GetCanonical() const -> BlockIdType {
    return id_ == source_id_ ? id_ : (file_->*ValueStore)().MakeCanonical(id_);
  }

  // Sets the element at index `i` within the block. Lazily allocates a new
  // block when the value changes for the first time.
  auto Set(int i, typename BlockIdType::ElementType value) -> void {
    if (source_id_.has_value() && (file_->*ValueStore)().Get(id_)[i] == value) {
      return;
    }
    if (id_ == source_id_) {
      id_ = (file_->*ValueStore)().Add((file_->*ValueStore)().Get(source_id_));
    }
    (file_->*ValueStore)().GetMutable(id_)[i] = value;
  }

 private:
  SemIR::File* file_;
  BlockIdType source_id_;
  BlockIdType id_ = source_id_;
};

using CopyOnWriteInstBlock = CopyOnWriteBlock<InstBlockId, &File::inst_blocks>;
using CopyOnWriteStructTypeFieldsBlock =
    CopyOnWriteBlock<StructTypeFieldsId, &File::struct_type_fields>;
using CopyOnWriteTypeBlock = CopyOnWriteBlock<TypeBlockId, &File::type_blocks>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_COPY_ON_WRITE_BLOCK_H_
