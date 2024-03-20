// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CONSTANT_H_
#define CARBON_TOOLCHAIN_SEM_IR_CONSTANT_H_

#include "llvm/ADT/FoldingSet.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::SemIR {

// Provides a ValueStore wrapper for tracking the constant values of
// instructions.
class ConstantValueStore {
 public:
  explicit ConstantValueStore(ConstantId default_value)
      : default_(default_value) {}

  // Returns the constant value of the requested instruction, which is default_
  // if unallocated.
  auto Get(InstId inst_id) const -> ConstantId {
    CARBON_CHECK(inst_id.index >= 0);
    return static_cast<size_t>(inst_id.index) >= values_.size()
               ? default_
               : values_[inst_id.index];
  }

  // Sets the constant value of the given instruction, or sets that it is known
  // to not be a constant.
  auto Set(InstId inst_id, ConstantId const_id) -> void {
    CARBON_CHECK(inst_id.index >= 0);
    if (static_cast<size_t>(inst_id.index) >= values_.size()) {
      values_.resize(inst_id.index + 1, default_);
    }
    values_[inst_id.index] = const_id;
  }

 private:
  const ConstantId default_;

  // A mapping from `InstId::index` to the corresponding constant value. This is
  // expected to be sparse, and may be smaller than the list of instructions if
  // there are trailing non-constant instructions.
  //
  // Set inline size to 0 because these will typically be too large for the
  // stack, while this does make File smaller.
  llvm::SmallVector<ConstantId, 0> values_;
};

// Provides storage for instructions representing deduplicated global constants.
class ConstantStore {
 public:
  explicit ConstantStore(File& sem_ir, llvm::BumpPtrAllocator& allocator)
      : allocator_(&allocator), constants_(&sem_ir) {}

  // Adds a new constant instruction, or gets the existing constant with this
  // value. Returns the ID of the constant.
  //
  // This updates `sem_ir.insts()` and `sem_ir.constant_values()` if the
  // constant is new.
  auto GetOrAdd(Inst inst, bool is_symbolic) -> ConstantId;

  // Returns a copy of the constant IDs as a vector, in an arbitrary but
  // stable order. This should not be used anywhere performance-sensitive.
  auto GetAsVector() const -> llvm::SmallVector<InstId, 0>;

  auto size() const -> int { return constants_.size(); }

 private:
  // TODO: We store two copies of each constant instruction: one in insts() and
  // one here. We could avoid one of those copies and store just an InstId here,
  // at the cost of some more indirection when recomputing profiles during
  // lookup. Once we have a representative data set, we should measure the
  // impact on compile time from that change.
  struct ConstantNode : llvm::FoldingSetNode {
    Inst inst;
    ConstantId constant_id;

    auto Profile(llvm::FoldingSetNodeID& id, File* sem_ir) -> void;
  };

  llvm::BumpPtrAllocator* allocator_;
  llvm::ContextualFoldingSet<ConstantNode, File*> constants_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CONSTANT_H_
