// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/subst.h"

#include "toolchain/sem_ir/ids.h"
#include "toolchain/check/eval.h"

namespace Carbon::Check {

namespace {

struct WorklistItem {
  // The instruction that we are substituting into.
  SemIR::InstId inst_id;
  // Whether the operands of this instruction have been added to the worklist.
  bool is_expanded : 1;
  // The index of the worklist item to process after we finish updating this
  // one. For the final child of an instruction, this is the parent. For any
  // other child, this is the index of the next child of the parent. For the
  // root, this is -1.
  int next_index : 31;
};

class Worklist {
 public:
  auto Get(int index) -> WorklistItem& { return worklist[index]; }
  auto Size() -> int { return worklist.size(); }
  auto Push(SemIR::InstId inst_id) -> void {
    worklist.push_back({.inst_id = inst_id,
                        .is_expanded = false,
                        .next_index = static_cast<int>(worklist.size() + 1)});
    CARBON_CHECK(worklist.back().next_index > 0) << "Constant too large.";
  }
  auto Pop() -> SemIR::InstId { return worklist.pop_back_val().inst_id; }

 private:
  // Constants can get pretty large, so use a large worklist. This should be
  // about 4KiB, which should be small enough to comfortably fit on the stack,
  // but large enough that it's unlikely that we'll need a heap allocation.
  llvm::SmallVector<WorklistItem, 512> worklist;
};

enum class ArgKind : std::uint8_t {
  Unimportant,
  InstId,
  TypeId,
  InstBlockId,
  TypeBlockId,
};

}  // namespace

// TODO: Move this into SemIR and refactor with `inst_profile.cpp`.
template<typename InstT, int N>
static constexpr auto GetArgKind() -> ArgKind {
  if constexpr (N >= SemIR::Internal::InstLikeTypeInfo<InstT>::NumArgs) {
    // This argument is not used by this instruction; don't profile it.
    return ArgKind::Unimportant;
  } else {
    using ArgT = SemIR::Internal::InstLikeTypeInfo<InstT>::template ArgType<N>;
    if (std::is_same_v<ArgT, SemIR::InstId>) {
      return ArgKind::InstId;
    } else if (std::is_same_v<ArgT, SemIR::TypeId>) {
      return ArgKind::TypeId;
    } else if (std::is_same_v<ArgT, SemIR::InstBlockId>) {
      return ArgKind::InstBlockId;
    } else if (std::is_same_v<ArgT, SemIR::TypeBlockId>) {
      return ArgKind::TypeBlockId;
    } else {
      return ArgKind::Unimportant;
    }
  }
}

static constexpr std::pair<ArgKind, ArgKind> ArgKinds[] = {
#define CARBON_SEM_IR_INST_KIND(KindName) \
  {GetArgKind<SemIR::KindName, 0>(), GetArgKind<SemIR::KindName, 1>()},
#include "toolchain/sem_ir/inst_kind.def"
};

// Pushes the specified operand onto the worklist.
static auto PushOperand(Context& context, Worklist& worklist, ArgKind kind,
                        int32_t arg) -> void {
  switch (kind) {
    case ArgKind::Unimportant:
      break;
    case ArgKind::InstId:
      worklist.Push(static_cast<SemIR::InstId>(arg));
      break;
    case ArgKind::TypeId:
      worklist.Push(context.types().GetInstId(static_cast<SemIR::TypeId>(arg)));
      break;
    case ArgKind::InstBlockId:
      for (auto inst_id :
           context.inst_blocks().Get(static_cast<SemIR::InstBlockId>(arg))) {
        worklist.Push(inst_id);
      }
      break;
    case ArgKind::TypeBlockId:
      for (auto type_id : context.type_blocks().Get(static_cast<SemIR::TypeBlockId>(arg))) {
        worklist.Push(context.types().GetInstId(type_id));
      }
      break;
  }
}

// Converts the operands of this instruction into `InstId`s and pushes them onto
// the worklist.
static auto ExpandOperands(Context& context, Worklist& worklist,
                           SemIR::InstId inst_id) -> void {
  auto inst = context.insts().Get(inst_id);
  auto kinds = ArgKinds[inst.kind().AsInt()];
  PushOperand(context, worklist, kinds.first, inst.arg0());
  PushOperand(context, worklist, kinds.second, inst.arg1());
}

// TODO: Copied from convert.cpp. Deduplicate!
namespace {
// A handle to a new block that may be modified, with copy-on-write semantics.
//
// The constructor is given the ID of an existing block that provides the
// initial contents of the new block. The new block is lazily allocated; if no
// modifications have been made, the `id()` function will return the original
// block ID.
//
// This is intended to avoid an unnecessary block allocation in the case where
// the new block ends up being exactly the same as the original block.
template<typename BlockIdType, auto (SemIR::File::*ValueStore)()>
class CopyOnWriteBlock {
 public:
  // Constructs the block. If `source_id` is valid, it is used as the initial
  // value of the block. Otherwise, uninitialized storage for `size` elements
  // is allocated.
  CopyOnWriteBlock(SemIR::File& file, BlockIdType source_id)
      : file_(file), source_id_(source_id) {}

  auto id() const -> BlockIdType { return id_; }

  auto Set(int i, typename BlockIdType::ElementType value) -> void {
    if (source_id_.is_valid() && (file_.*ValueStore)().Get(id_)[i] == value) {
      return;
    }
    if (id_ == source_id_) {
      id_ = (file_.*ValueStore)().Add((file_.*ValueStore)().Get(source_id_));
    }
    (file_.*ValueStore)().Get(id_)[i] = value;
  }

 private:
  SemIR::File& file_;
  BlockIdType source_id_;
  BlockIdType id_ = source_id_;
};
}  // namespace

// Pops the specified operand from the worklist and returns it.
static auto PopOperand(Context& context, Worklist& worklist, ArgKind kind,
                       int32_t arg) -> int32_t {
  switch (kind) {
    case ArgKind::Unimportant:
      return arg;
    case ArgKind::InstId:
      return worklist.Pop().index;
    case ArgKind::TypeId:
      return context.GetTypeIdForTypeInst(worklist.Pop()).index;
    case ArgKind::InstBlockId: {
      auto old_inst_block_id = static_cast<SemIR::InstBlockId>(arg);
      auto size = context.inst_blocks().Get(old_inst_block_id).size();
      CopyOnWriteBlock<SemIR::InstBlockId, &SemIR::File::inst_blocks>
          new_inst_block(context.sem_ir(), old_inst_block_id);
      for (auto i : llvm::index_range(0, size)) {
        new_inst_block.Set(size - i - 1, worklist.Pop());
      }
      return new_inst_block.id().index;
    }
    case ArgKind::TypeBlockId: {
      auto old_type_block_id = static_cast<SemIR::TypeBlockId>(arg);
      auto size = context.type_blocks().Get(old_type_block_id).size();
      CopyOnWriteBlock<SemIR::TypeBlockId, &SemIR::File::type_blocks>
          new_type_block(context.sem_ir(), old_type_block_id);
      for (auto i : llvm::index_range(0, size)) {
        new_type_block.Set(size - i - 1,
                           context.GetTypeIdForTypeInst(worklist.Pop()));
      }
      return new_type_block.id().index;
    }
  }
}

// Pops the operands of the specified instruction off the worklist and rebuilds
// the instruction with the updated operands.
static auto Rebuild(Context& context, Worklist& worklist,
                    SemIR::InstId inst_id) -> SemIR::InstId {
  auto inst = context.insts().Get(inst_id);
  auto kinds = ArgKinds[inst.kind().AsInt()];

  // Note that we pop in reverse order because we pushed them in forwards order.
  int32_t arg1 = PopOperand(context, worklist, kinds.second, inst.arg1());
  int32_t arg0 = PopOperand(context, worklist, kinds.first, inst.arg0());
  if (arg0 == inst.arg0() && arg1 == inst.arg1()) {
    return inst_id;
  }

  inst.SetArgs(arg0, arg1);
  auto result_id = TryEvalInst(context, SemIR::InstId::Invalid, inst);
  CARBON_CHECK(result_id.is_constant())
      << "Substitution into constant produced non-constant";
  return result_id.inst_id();
}

auto SubstConstant(Context& context, SemIR::ConstantId const_id,
                   Substitutions substitutions) -> SemIR::ConstantId {
  CARBON_CHECK(const_id.is_constant()) << "Substituting into non-constant";

  if (substitutions.empty()) {
    // Nothing to substitute.
    return const_id;
  }

  if (!const_id.is_symbolic()) {
    // A template constant can't contain a reference to a symbolic binding.
    return const_id;
  }

  Worklist worklist;
  worklist.Push(const_id.inst_id());
  worklist.Get(0).next_index = -1;

  // For each instruction that forms part of the constant, we will visit it
  // twice:
  //
  // - First, we visit it with `is_expanded == false`, we add all of its
  //   operands onto the worklist, and process them by following this same
  //   process.
  // - Then, once all operands are processed, we visit the instruction with
  //   `is_expanded == true`, pop the operands back off the worklist, and if any
  //   of them changed, rebuild this instruction.
  //
  // The second step is skipped if we can detect in the first step that the
  // instruction will not need to be rebuilt.
  int index = 0;
  while (index != -1) {
    auto& item = worklist.Get(index);
    if (!item.is_expanded) {
      if (context.constant_values().Get(item.inst_id).is_template()) {
        // This instruction is a template constant, so can't contain any
        // bindings that need to be substituted.
        index = item.next_index;
        continue;
      }

      if (context.insts().Is<SemIR::BindSymbolicName>(item.inst_id)) {
        // This is a symbolic binding. Check if we're substituting it.

        // TODO: Consider building a hash map for substitutions. We might have a
        // lot of them.
        for (auto [bind_id, replacement_id] : substitutions) {
          if (item.inst_id == bind_id) {
            // This is the binding we're replacing. Perform substitution.
            item.inst_id = replacement_id.inst_id();
            break;
          }
        }

        // Even if it's not being substituted, don't look through it. Its constant value doesn't spend on its operand.
        index = item.next_index;
        continue;
      }

      // Extract the operands of this item into the worklist. Note that this
      // modifies the worklist, so it's not safe to use `item` after
      // `ExpandOperands` returns.
      item.is_expanded = true;
      int first_operand = worklist.Size();
      int next_index = item.next_index;
      ExpandOperands(context, worklist, item.inst_id);

      // If there are any operands, go and update them before rebuilding this
      // item.
      if (worklist.Size() > first_operand) {
        worklist.Get(worklist.Size() - 1).next_index = index;
        index = first_operand;
      } else {
        // No need to rebuild this instruction.
        index = next_index;
      }
    } else {
      // Rebuild this item if necessary. Note that this might pop items from the
      // worklist but does not reallocate, so does not invalidate `item`.
      item.inst_id = Rebuild(context, worklist, item.inst_id);
      index = item.next_index;
    }
  }

  CARBON_CHECK(worklist.Size() == 1)
      << "Unexpected data left behind in work list";
  return context.constant_values().Get(worklist.Get(0).inst_id);
}

auto SubstType(Context& context, SemIR::TypeId type_id,
                   Substitutions substitutions) -> SemIR::TypeId {
  return context.GetTypeIdForTypeConstant(SubstConstant(
      context, context.types().GetConstantId(type_id), substitutions));
}

}  // namespace Carbon::Check
