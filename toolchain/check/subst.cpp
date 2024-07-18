// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/subst.h"

#include "toolchain/check/eval.h"
#include "toolchain/sem_ir/copy_on_write_block.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

namespace {

// Information about an instruction that we are substituting into.
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

// A list of instructions that we're currently in the process of substituting
// into. For details of the algorithm used here, see `SubstConstant`.
class Worklist {
 public:
  explicit Worklist(SemIR::InstId root_id) {
    worklist_.push_back(
        {.inst_id = root_id, .is_expanded = false, .next_index = -1});
  }

  auto operator[](int index) -> WorklistItem& { return worklist_[index]; }
  auto size() -> int { return worklist_.size(); }
  auto back() -> WorklistItem& { return worklist_.back(); }

  auto Push(SemIR::InstId inst_id) -> void {
    worklist_.push_back({.inst_id = inst_id,
                         .is_expanded = false,
                         .next_index = static_cast<int>(worklist_.size() + 1)});
    CARBON_CHECK(worklist_.back().next_index > 0) << "Constant too large.";
  }
  auto Pop() -> SemIR::InstId { return worklist_.pop_back_val().inst_id; }

 private:
  // Constants can get pretty large, so use a large worklist. This should be
  // about 4KiB, which should be small enough to comfortably fit on the stack,
  // but large enough that it's unlikely that we'll need a heap allocation.
  llvm::SmallVector<WorklistItem, 512> worklist_;
};

}  // namespace

// Pushes the specified operand onto the worklist.
static auto PushOperand(Context& context, Worklist& worklist,
                        SemIR::IdKind kind, int32_t arg) -> void {
  switch (kind) {
    case SemIR::IdKind::For<SemIR::InstId>:
      worklist.Push(SemIR::InstId(arg));
      break;
    case SemIR::IdKind::For<SemIR::TypeId>:
      if (auto type_id = SemIR::TypeId(arg); type_id.is_valid()) {
        worklist.Push(context.types().GetInstId(type_id));
      }
      break;
    case SemIR::IdKind::For<SemIR::InstBlockId>:
      for (auto inst_id : context.inst_blocks().Get(SemIR::InstBlockId(arg))) {
        worklist.Push(inst_id);
      }
      break;
    case SemIR::IdKind::For<SemIR::TypeBlockId>:
      for (auto type_id : context.type_blocks().Get(SemIR::TypeBlockId(arg))) {
        worklist.Push(context.types().GetInstId(type_id));
      }
      break;
    default:
      break;
  }
}

// Converts the operands of this instruction into `InstId`s and pushes them onto
// the worklist.
static auto ExpandOperands(Context& context, Worklist& worklist,
                           SemIR::InstId inst_id) -> void {
  auto inst = context.insts().Get(inst_id);
  auto kinds = inst.ArgKinds();
  PushOperand(context, worklist, SemIR::IdKind::For<SemIR::TypeId>,
              inst.type_id().index);
  PushOperand(context, worklist, kinds.first, inst.arg0());
  PushOperand(context, worklist, kinds.second, inst.arg1());
}

// Pops the specified operand from the worklist and returns it.
static auto PopOperand(Context& context, Worklist& worklist, SemIR::IdKind kind,
                       int32_t arg) -> int32_t {
  switch (kind) {
    case SemIR::IdKind::For<SemIR::InstId>:
      return worklist.Pop().index;
    case SemIR::IdKind::For<SemIR::TypeId>: {
      auto type_id = SemIR::TypeId(arg);
      if (!type_id.is_valid()) {
        return arg;
      }
      return context.GetTypeIdForTypeInst(worklist.Pop()).index;
    }
    case SemIR::IdKind::For<SemIR::InstBlockId>: {
      auto old_inst_block_id = SemIR::InstBlockId(arg);
      auto size = context.inst_blocks().Get(old_inst_block_id).size();
      SemIR::CopyOnWriteInstBlock new_inst_block(context.sem_ir(),
                                                 old_inst_block_id);
      for (auto i : llvm::reverse(llvm::seq(size))) {
        new_inst_block.Set(i, worklist.Pop());
      }
      return new_inst_block.GetCanonical().index;
    }
    case SemIR::IdKind::For<SemIR::TypeBlockId>: {
      auto old_type_block_id = SemIR::TypeBlockId(arg);
      auto size = context.type_blocks().Get(old_type_block_id).size();
      SemIR::CopyOnWriteTypeBlock new_type_block(context.sem_ir(),
                                                 old_type_block_id);
      for (auto i : llvm::index_range(0, size)) {
        new_type_block.Set(size - i - 1,
                           context.GetTypeIdForTypeInst(worklist.Pop()));
      }
      return new_type_block.GetCanonical().index;
    }
    default:
      return arg;
  }
}

// Pops the operands of the specified instruction off the worklist and rebuilds
// the instruction with the updated operands.
static auto Rebuild(Context& context, Worklist& worklist, SemIR::InstId inst_id)
    -> SemIR::InstId {
  auto inst = context.insts().Get(inst_id);
  auto kinds = inst.ArgKinds();

  // Note that we pop in reverse order because we pushed them in forwards order.
  int32_t arg1 = PopOperand(context, worklist, kinds.second, inst.arg1());
  int32_t arg0 = PopOperand(context, worklist, kinds.first, inst.arg0());
  int32_t type_id =
      PopOperand(context, worklist, SemIR::IdKind::For<SemIR::TypeId>,
                 inst.type_id().index);
  if (type_id == inst.type_id().index && arg0 == inst.arg0() &&
      arg1 == inst.arg1()) {
    return inst_id;
  }

  // TODO: Do we need to require this type to be complete?
  inst.SetType(SemIR::TypeId(type_id));
  inst.SetArgs(arg0, arg1);
  auto result_id = TryEvalInst(context, SemIR::InstId::Invalid, inst);
  CARBON_CHECK(result_id.is_constant())
      << "Substitution into constant produced non-constant";
  return context.constant_values().GetInstId(result_id);
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

  // TODO: Consider caching; we may perform the same substitutions repeatedly.
  Worklist worklist(context.constant_values().GetInstId(const_id));

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
    auto& item = worklist[index];

    if (item.is_expanded) {
      // Rebuild this item if necessary. Note that this might pop items from the
      // worklist but does not reallocate, so does not invalidate `item`.
      item.inst_id = Rebuild(context, worklist, item.inst_id);
      index = item.next_index;
      continue;
    }

    if (context.constant_values().Get(item.inst_id).is_template()) {
      // This instruction is a template constant, so can't contain any
      // bindings that need to be substituted.
      index = item.next_index;
      continue;
    }

    if (auto bind =
            context.insts().TryGetAs<SemIR::BindSymbolicName>(item.inst_id)) {
      // This is a symbolic binding. Check if we're substituting it.

      // TODO: Consider building a hash map for substitutions. We might have a
      // lot of them.
      for (auto [bind_index, replacement_id] : substitutions) {
        if (context.entity_names().Get(bind->entity_name_id).bind_index ==
            bind_index) {
          // This is the binding we're replacing. Perform substitution.
          item.inst_id = context.constant_values().GetInstId(replacement_id);
          break;
        }
      }

      // If it's not being substituted, don't look through it. Its constant
      // value doesn't depend on its operand.
      index = item.next_index;
      continue;
    }

    // Extract the operands of this item into the worklist. Note that this
    // modifies the worklist, so it's not safe to use `item` after
    // `ExpandOperands` returns.
    item.is_expanded = true;
    int first_operand = worklist.size();
    int next_index = item.next_index;
    ExpandOperands(context, worklist, item.inst_id);

    // If there are any operands, go and update them before rebuilding this
    // item.
    if (worklist.size() > first_operand) {
      worklist.back().next_index = index;
      index = first_operand;
    } else {
      // No need to rebuild this instruction.
      index = next_index;
    }
  }

  CARBON_CHECK(worklist.size() == 1)
      << "Unexpected data left behind in work list";
  return context.constant_values().Get(worklist.back().inst_id);
}

auto SubstType(Context& context, SemIR::TypeId type_id,
               Substitutions substitutions) -> SemIR::TypeId {
  return context.GetTypeIdForTypeConstant(SubstConstant(
      context, context.types().GetConstantId(type_id), substitutions));
}

}  // namespace Carbon::Check
