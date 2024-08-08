// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/subst.h"

#include "toolchain/check/eval.h"
#include "toolchain/check/generic.h"
#include "toolchain/sem_ir/copy_on_write_block.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

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
    CARBON_CHECK(inst_id.is_valid());
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
      if (auto inst_id = SemIR::InstId(arg); inst_id.is_valid()) {
        worklist.Push(inst_id);
      }
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
    case SemIR::IdKind::For<SemIR::SpecificId>:
      if (auto specific_id = static_cast<SemIR::SpecificId>(arg);
          specific_id.is_valid()) {
        PushOperand(context, worklist, SemIR::IdKind::For<SemIR::InstBlockId>,
                    context.specifics().Get(specific_id).args_id.index);
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
    case SemIR::IdKind::For<SemIR::InstId>: {
      auto inst_id = SemIR::InstId(arg);
      if (!inst_id.is_valid()) {
        return arg;
      }
      return worklist.Pop().index;
    }
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
    case SemIR::IdKind::For<SemIR::SpecificId>: {
      auto specific_id = SemIR::SpecificId(arg);
      if (!specific_id.is_valid()) {
        return arg;
      }
      auto& specific = context.specifics().Get(specific_id);
      auto args_id =
          PopOperand(context, worklist, SemIR::IdKind::For<SemIR::InstBlockId>,
                     specific.args_id.index);
      return MakeSpecific(context, specific.generic_id,
                          SemIR::InstBlockId(args_id))
          .index;
    }
    default:
      return arg;
  }
}

// Pops the operands of the specified instruction off the worklist and rebuilds
// the instruction with the updated operands if it has changed.
static auto Rebuild(Context& context, Worklist& worklist, SemIR::InstId inst_id,
                    const SubstInstCallbacks& callbacks) -> SemIR::InstId {
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
  return callbacks.Rebuild(inst_id, inst);
}

auto SubstInst(Context& context, SemIR::InstId inst_id,
               const SubstInstCallbacks& callbacks) -> SemIR::InstId {
  Worklist worklist(inst_id);

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
      item.inst_id = Rebuild(context, worklist, item.inst_id, callbacks);
      index = item.next_index;
      continue;
    }

    if (callbacks.Subst(item.inst_id)) {
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
      // No need to rebuild this instruction: its operands can't be changed by
      // substitution because it has none.
      index = next_index;
    }
  }

  CARBON_CHECK(worklist.size() == 1)
      << "Unexpected data left behind in work list";
  return worklist.back().inst_id;
}

namespace {
// Callbacks for performing substitution of a set of Substitutions into a
// symbolic constant.
class SubstConstantCallbacks final : public SubstInstCallbacks {
 public:
  SubstConstantCallbacks(Context& context, Substitutions substitutions)
      : context_(context), substitutions_(substitutions) {}

  // Applies the given Substitutions to an instruction, in order to replace
  // BindSymbolicName instructions with the value of the binding.
  auto Subst(SemIR::InstId& inst_id) const -> bool override {
    if (context_.constant_values().Get(inst_id).is_template()) {
      // This instruction is a template constant, so can't contain any
      // bindings that need to be substituted.
      return true;
    }

    auto bind = context_.insts().TryGetAs<SemIR::BindSymbolicName>(inst_id);
    if (!bind) {
      return false;
    }

    // This is a symbolic binding. Check if we're substituting it.
    // TODO: Consider building a hash map for substitutions. We might have a
    // lot of them.
    for (auto [bind_index, replacement_id] : substitutions_) {
      if (context_.entity_names().Get(bind->entity_name_id).bind_index ==
          bind_index) {
        // This is the binding we're replacing. Perform substitution.
        inst_id = context_.constant_values().GetInstId(replacement_id);
        return true;
      }
    }

    // If it's not being substituted, don't look through it. Its constant
    // value doesn't depend on its operand.
    return true;
  }

  // Rebuilds an instruction by building a new constant.
  auto Rebuild(SemIR::InstId /*old_inst_id*/, SemIR::Inst new_inst) const
      -> SemIR::InstId override {
    auto result_id = TryEvalInst(context_, SemIR::InstId::Invalid, new_inst);
    CARBON_CHECK(result_id.is_constant())
        << "Substitution into constant produced non-constant";
    return context_.constant_values().GetInstId(result_id);
  }

 private:
  Context& context_;
  Substitutions substitutions_;
};
}  // namespace

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

  auto subst_inst_id =
      SubstInst(context, context.constant_values().GetInstId(const_id),
                SubstConstantCallbacks(context, substitutions));
  return context.constant_values().Get(subst_inst_id);
}

}  // namespace Carbon::Check
