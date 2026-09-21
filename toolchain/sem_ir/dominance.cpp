// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/dominance.h"

#include <algorithm>
#include <utility>

#include "common/check.h"
#include "common/error.h"
#include "common/map.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/grouped_value_store.h"
#include "toolchain/base/id_tag.h"
#include "toolchain/base/index_base.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/id_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {
namespace {

// The position of a block within a function's list of body blocks. This is used
// to index the control flow graph and dominator tree built for the function.
struct BlockIndex : public IndexBase<BlockIndex> {
  // Only used when a `BlockIndex` is printed, which is useful when debugging.
  [[maybe_unused]] static constexpr llvm::StringLiteral Label = "block";

  using IndexBase::IndexBase;
};

// Execution of a function body starts in its first block.
constexpr BlockIndex EntryBlockIndex(0);

// A step in the walk over a function's dominator tree.
struct WalkStep {
  // Returns a step that verifies the instructions in `block_index` and queues
  // up the blocks it dominates.
  static auto EnterBlock(BlockIndex block_index) -> WalkStep {
    return {.block_index = block_index, .scope_start = 0};
  }

  // Returns a step that leaves the scope of a block that was entered when the
  // list of evaluated instructions had size `scope_start`.
  static auto LeaveBlock(int scope_start) -> WalkStep {
    return {.block_index = BlockIndex::None, .scope_start = scope_start};
  }

  // The block to enter, or `None` if this step leaves a block instead.
  BlockIndex block_index;
  // The number of instructions that had been evaluated when the block being
  // left was entered. Only used when leaving a block.
  int scope_start;
};

// Returns the block that `inst` transfers control to, or `InstBlockId::None` if
// `inst` is not a branch.
auto GetBranchTargetId(Inst inst) -> InstBlockId {
  if (auto branch = inst.TryAs<AnyBranch>()) {
    return branch->target_id;
  }
  return InstBlockId::None;
}

// A set of the instructions in a file, holding one bit per instruction.
//
// Creating one of these costs a bit per instruction in the file, so they are
// created once for the file rather than once per function.
class InstSet {
 public:
  explicit InstSet(const File& file)
      : tag_(file.insts().GetIdTag()), bits_(file.insts().size()) {}

  // Adds `inst_id`, returning whether it wasn't already present.
  auto Insert(InstId inst_id) -> bool {
    int32_t index = tag_.Remove(inst_id);
    if (bits_.test(index)) {
      return false;
    }
    bits_.set(index);
    return true;
  }

  auto Erase(InstId inst_id) -> void { bits_.reset(tag_.Remove(inst_id)); }

  auto Contains(InstId inst_id) const -> bool {
    return bits_.test(tag_.Remove(inst_id));
  }

 private:
  // Instruction IDs are tagged, so the tag is removed to recover the index of
  // the instruction, which is the index of its bit.
  InstStore::IdTagType tag_;
  llvm::BitVector bits_;
};

// Collects the instructions that are referenced from function bodies despite
// being evaluated at file scope, or not evaluated at all, into `decl_insts`.
//
// TODO: These are pre-existing dominance violations, not genuine exemptions:
//
// -   File-scope instructions are evaluated in `__global_init`, if at all, so
//     they don't dominate uses in another function.
// -   A `let` in a class body produces a `wrapper_binding` that isn't evaluated
//     in any function, but a qualified name reference to it can appear in one.
//     For example, from the `public_global_access` case in
//     `check/testdata/class/access/access_modifiers.carbon`:
//
//         class A { let x: i32 = 5; }
//         let x: i32 = A.x;
//
//     Here the `name_ref` for `A.x` is in `__global_init`, and the
//     `wrapper_binding` for `x` is only in `A`'s body block.
//
// Lowering only works today because such uses either happen to be constant or
// are never lowered. Remove this allowlist and diagnose the uses instead once
// global initialization semantics and the member reference model are settled.
auto CollectDeclInsts(const File& file, InstSet& decl_insts) -> void {
  auto add_block = [&](InstBlockId block_id) {
    if (block_id.has_value()) {
      for (InstId inst_id : file.inst_blocks().Get(block_id)) {
        if (inst_id.has_value()) {
          decl_insts.Insert(inst_id);
        }
      }
    }
  };

  add_block(file.top_inst_block_id());
  for (const auto& class_info : file.classes().values()) {
    add_block(class_info.body_block_id);
  }
}

// The specifics of each generic in a file.
using SpecificsByGeneric =
    GroupedValueStore<GenericId, SpecificId, Tag<CheckIRId>>;

// Collects the specifics in `file`, grouped by the generic they're a specific
// of. A specific that has never been resolved, or whose resolution failed,
// doesn't have instructions to check, so is omitted.
//
// This grouping is built once for the file so that each generic function can
// find its own specifics without scanning all of them.
auto CollectSpecifics(const File& file) -> SpecificsByGeneric {
  return SpecificsByGeneric(file.generics(), [&](auto add) {
    for (const auto& [specific_id, specific] : file.specifics().enumerate()) {
      if (specific.IsUnresolved() || specific.HasError()) {
        continue;
      }
      add(specific.generic_id, specific_id);
    }
  });
}

// The dominator tree of a function body: the blocks that each block
// immediately dominates, indexed by `BlockIndex`.
using DominatorTree = llvm::SmallVector<llvm::SmallVector<BlockIndex, 2>>;

// Builds the control flow graph of a function body and its dominator tree.
//
// These depend only on the branches between the function's body blocks, which
// are the same in every specific of a generic function: a block spliced into
// the body never contains control flow. So the tree is built once per function
// and shared by the verification of each of its specifics.
class DominatorTreeBuilder {
 public:
  explicit DominatorTreeBuilder(const File& file, const Function& function)
      : file_(file),
        function_(function),
        body_blocks_(function.body_block_ids) {}

  // Returns the dominator tree of the function body, diagnosing a branch that
  // leaves the body and blocks that are unreachable from the entry block.
  auto Build() -> ErrorOr<DominatorTree>;

 private:
  // Builds `successors_` and `predecessors_` from the branches in each block.
  auto BuildControlFlowGraph() -> ErrorOr<Success>;

  // Appends the blocks reachable from the entry block to `post_order`, in
  // post-order, and marks each of them `visited`.
  auto BuildPostOrder(llvm::SmallBitVector& visited,
                      llvm::SmallVectorImpl<BlockIndex>& post_order) -> void;

  auto num_blocks() const -> int { return body_blocks_.size(); }

  const File& file_;
  const Function& function_;
  llvm::ArrayRef<InstBlockId> body_blocks_;

  // The index of each block in `body_blocks_`.
  Map<InstBlockId, BlockIndex> block_indexes_;

  // The control flow graph, indexed by `BlockIndex`.
  llvm::SmallVector<llvm::SmallVector<BlockIndex, 2>> successors_;
  llvm::SmallVector<llvm::SmallVector<BlockIndex, 2>> predecessors_;
};

// Verifies that every operand of every instruction in one function body is
// dominated by an evaluation of that operand.
//
// This walks the dominator tree of the function's control flow graph, tracking
// the set of instructions whose evaluations dominate the point currently being
// checked. An instruction joins that set when its evaluation is reached, and
// leaves it again when the walk leaves the blocks that the evaluation
// dominates.
//
// TODO: Improve LLVM's GenericDomTree implementation so that it's compatible
// with our graph representation, then rewrite this to use that rather than
// implementing our own dominator tree construction. Currently, GenericDomTree
// requires a pointer-based data structure, and building such a data structure
// introduces a substantial performance overhead compared to running dominator
// tree construction directly on our SemIR representation.
class DominanceVerifier {
 public:
  explicit DominanceVerifier(const File& file, const InstSet& decl_insts,
                             const Function& function,
                             const DominatorTree& dom_tree, InstSet& evaluated,
                             SpecificId specific_id)
      : file_(file),
        decl_insts_(decl_insts),
        function_(function),
        dom_tree_(dom_tree),
        specific_id_(specific_id),
        body_blocks_(function.body_block_ids),
        evaluated_(evaluated) {}

  auto Verify() -> ErrorOr<Success>;

 private:
  // Verifies every block, walking the dominator tree from the entry block.
  auto VerifyBlocks() -> ErrorOr<Success>;

  // Verifies the operands of `inst_id`, then records `inst_id`, along with any
  // instructions it splices into the enclosing block, as evaluated.
  auto VerifyAndRecordInst(InstId root_inst_id, BlockIndex block_index)
      -> ErrorOr<Success>;

  // Verifies a single operand of `user_id`.
  auto VerifyArg(InstId user_id, IdAndKind arg, BlockIndex block_index)
      -> ErrorOr<Success>;

  // Verifies that `operand_id`, used by `user_id` in `block_index`, is either
  // constant or dominated by an evaluation.
  auto VerifyOperand(InstId user_id, InstId operand_id, BlockIndex block_index)
      -> ErrorOr<Success>;

  // Records that `inst_id` is evaluated at the point currently being verified.
  auto RecordEvaluated(InstId inst_id) -> void;

  // Records that every instruction in `block_id`, if it has a value, is
  // evaluated at the point currently being verified.
  auto RecordEvaluatedBlock(InstBlockId block_id) -> void;

  // Returns the instruction that `splice` splices in, or `InstId::None` if that
  // can't be determined.
  auto GetSplicedInstId(SpliceInst splice) const -> InstId;

  const File& file_;
  const InstSet& decl_insts_;
  const Function& function_;
  const DominatorTree& dom_tree_;
  SpecificId specific_id_;
  llvm::ArrayRef<InstBlockId> body_blocks_;

  // The instructions whose evaluations dominate the point currently being
  // verified, and the order in which they were added, so that they can be
  // removed again when leaving a block.
  //
  // `evaluated_` is owned by the caller and shared between functions, because
  // creating one costs a bit per instruction in the whole file. `Verify` leaves
  // it empty.
  InstSet& evaluated_;
  llvm::SmallVector<InstId> evaluated_order_;
};

auto DominanceVerifier::Verify() -> ErrorOr<Success> {
  CARBON_CHECK(!body_blocks_.empty());

  // Parameters and other instructions from the function declaration are
  // evaluated before the body begins, so they dominate the whole body.
  RecordEvaluatedBlock(function_.call_params_id);
  RecordEvaluatedBlock(function_.call_param_patterns_id);
  RecordEvaluatedBlock(function_.call_param_default_values_id);
  RecordEvaluatedBlock(function_.pattern_block_id);
  RecordEvaluated(function_.self_param_id);
  RecordEvaluated(function_.return_form_inst_id);
  RecordEvaluated(function_.return_pattern_id);
  for (InstId decl_id :
       {function_.definition_id, function_.first_owning_decl_id,
        function_.non_owning_decl_id}) {
    if (!decl_id.has_value()) {
      continue;
    }
    if (auto decl = file_.insts().TryGetAs<FunctionDecl>(decl_id)) {
      RecordEvaluatedBlock(decl->decl_block_id);
    }
  }

  auto result = VerifyBlocks();

  // `evaluated_` is shared with the verification of other functions, so put it
  // back the way it was found. Every instruction this added to it is in
  // `evaluated_order_`, including if `VerifyBlocks` stopped at an error.
  for (InstId inst_id : evaluated_order_) {
    evaluated_.Erase(inst_id);
  }
  evaluated_order_.clear();

  return result;
}

auto DominatorTreeBuilder::BuildControlFlowGraph() -> ErrorOr<Success> {
  for (int i = 0; i != num_blocks(); ++i) {
    block_indexes_.Insert(body_blocks_[i], BlockIndex(i));
  }

  successors_.resize(num_blocks());
  predecessors_.resize(num_blocks());
  for (int i = 0; i != num_blocks(); ++i) {
    BlockIndex from(i);
    for (InstId inst_id : file_.inst_blocks().Get(body_blocks_[i])) {
      InstBlockId target_id = GetBranchTargetId(file_.insts().Get(inst_id));
      if (!target_id.has_value()) {
        continue;
      }
      BlockIndex* to = block_indexes_[target_id];
      if (!to) {
        return ErrorBuilder()
               << "Branch in block " << body_blocks_[i] << " targets block "
               << target_id << " which is not in function body";
      }
      // A block that branches to the same target more than once produces a
      // duplicate edge. That's harmless: the post-order walk skips blocks it
      // has already visited, and intersecting the dominators of the same
      // predecessor twice gives the same result. Removing duplicates here
      // would instead be quadratic in a block's number of successors.
      successors_[from.index].push_back(*to);
      predecessors_[to->index].push_back(from);
    }
  }
  return Success();
}

auto DominatorTreeBuilder::BuildPostOrder(
    llvm::SmallBitVector& visited,
    llvm::SmallVectorImpl<BlockIndex>& post_order) -> void {
  // The blocks whose successors are still being visited, each paired with the
  // number of its successors that have been visited so far. A block is appended
  // to `post_order` once all of its successors have been visited.
  llvm::SmallVector<std::pair<BlockIndex, int>> stack;
  visited.set(EntryBlockIndex.index);
  stack.push_back({EntryBlockIndex, 0});

  while (!stack.empty()) {
    auto [block_index, num_visited] = stack.back();
    const auto& successors = successors_[block_index.index];
    if (num_visited == static_cast<int>(successors.size())) {
      post_order.push_back(block_index);
      stack.pop_back();
      continue;
    }

    stack.back().second = num_visited + 1;
    BlockIndex successor = successors[num_visited];
    if (!visited.test(successor.index)) {
      visited.set(successor.index);
      stack.push_back({successor, 0});
    }
  }
}

auto DominatorTreeBuilder::Build() -> ErrorOr<DominatorTree> {
  CARBON_CHECK(!body_blocks_.empty());
  CARBON_RETURN_IF_ERROR(BuildControlFlowGraph());

  // Order the blocks so that, apart from loop back edges, every block precedes
  // its successors.
  llvm::SmallBitVector visited(num_blocks());
  llvm::SmallVector<BlockIndex> reverse_post_order;
  reverse_post_order.reserve(num_blocks());
  BuildPostOrder(visited, reverse_post_order);
  std::reverse(reverse_post_order.begin(), reverse_post_order.end());

  for (int i = 0; i != num_blocks(); ++i) {
    if (!visited.test(i)) {
      return ErrorBuilder()
             << "Block " << body_blocks_[i] << " in function "
             << function_.name_id << " is unreachable from entry block";
    }
  }

  llvm::SmallVector<int> order(num_blocks(), -1);
  for (int i = 0; i != num_blocks(); ++i) {
    order[reverse_post_order[i].index] = i;
  }

  // Find the immediate dominator of each block using the algorithm from "A
  // Simple, Fast Dominance Algorithm" by Cooper, Harvey, and Kennedy:
  // repeatedly sweep the blocks in reverse post-order, intersecting the
  // dominators of each block's predecessors, until the result stops changing.
  llvm::SmallVector<BlockIndex> idom(num_blocks(), BlockIndex::None);
  idom[EntryBlockIndex.index] = EntryBlockIndex;

  // Returns the closest block that dominates both `lhs` and `rhs`, by walking
  // both up the dominator tree built so far until they meet.
  auto intersect = [&](BlockIndex lhs, BlockIndex rhs) -> BlockIndex {
    while (lhs != rhs) {
      while (order[lhs.index] > order[rhs.index]) {
        lhs = idom[lhs.index];
      }
      while (order[rhs.index] > order[lhs.index]) {
        rhs = idom[rhs.index];
      }
    }
    return lhs;
  };

  for (bool changed = true; changed;) {
    changed = false;
    // The entry block is its own immediate dominator, so start after it.
    for (BlockIndex block_index : llvm::drop_begin(reverse_post_order)) {
      BlockIndex new_idom = BlockIndex::None;
      for (BlockIndex predecessor : predecessors_[block_index.index]) {
        if (!idom[predecessor.index].has_value()) {
          continue;
        }
        new_idom = new_idom.has_value() ? intersect(predecessor, new_idom)
                                        : predecessor;
      }
      if (new_idom.has_value() && idom[block_index.index] != new_idom) {
        idom[block_index.index] = new_idom;
        changed = true;
      }
    }
  }

  DominatorTree dom_children(num_blocks());
  for (BlockIndex block_index : llvm::drop_begin(reverse_post_order)) {
    // Every block other than the entry block is reachable, and so is dominated
    // by the predecessor it's reached through.
    CARBON_CHECK(idom[block_index.index].has_value(), "No dominator for {0}",
                 body_blocks_[block_index.index]);
    dom_children[idom[block_index.index].index].push_back(block_index);
  }
  return dom_children;
}

auto DominanceVerifier::VerifyBlocks() -> ErrorOr<Success> {
  llvm::SmallVector<WalkStep> worklist = {
      WalkStep::EnterBlock(EntryBlockIndex)};

  while (!worklist.empty()) {
    auto [block_index, scope_start] = worklist.pop_back_val();

    if (!block_index.has_value()) {
      for (InstId inst_id : llvm::drop_begin(evaluated_order_, scope_start)) {
        evaluated_.Erase(inst_id);
      }
      evaluated_order_.truncate(scope_start);
      continue;
    }

    // Evaluations in this block dominate the rest of this block and the blocks
    // below it in the dominator tree, but nothing else. This step is beneath
    // this block's children on the worklist, so it runs once they're done.
    worklist.push_back(WalkStep::LeaveBlock(evaluated_order_.size()));

    for (InstId inst_id :
         file_.inst_blocks().Get(body_blocks_[block_index.index])) {
      CARBON_RETURN_IF_ERROR(VerifyAndRecordInst(inst_id, block_index));
    }
    for (BlockIndex child : dom_tree_[block_index.index]) {
      worklist.push_back(WalkStep::EnterBlock(child));
    }
  }
  return Success();
}

auto DominanceVerifier::VerifyAndRecordInst(InstId root_inst_id,
                                            BlockIndex block_index)
    -> ErrorOr<Success> {
  struct Step {
    InstId inst_id;
    // Whether to finish a `SpliceBlock` after its block has been evaluated.
    bool finish_splice_block = false;
  };

  llvm::SmallVector<Step> worklist = {{.inst_id = root_inst_id}};
  while (!worklist.empty()) {
    auto [inst_id, finish_splice_block] = worklist.pop_back_val();
    if (finish_splice_block) {
      auto splice_block = file_.insts().GetAs<SpliceBlock>(inst_id);
      CARBON_RETURN_IF_ERROR(
          VerifyOperand(inst_id, splice_block.result_id, block_index));
      RecordEvaluated(inst_id);
      continue;
    }

    Inst inst = file_.insts().Get(inst_id);

    // A `SpliceBlock` evaluates the instructions in its block, and then
    // produces the value of one of them.
    if (auto splice_block = inst.TryAs<SpliceBlock>()) {
      worklist.push_back({.inst_id = inst_id, .finish_splice_block = true});
      if (splice_block->block_id.has_value()) {
        for (InstId spliced_id :
             llvm::reverse(file_.inst_blocks().Get(splice_block->block_id))) {
          worklist.push_back({.inst_id = spliced_id});
        }
      }
      continue;
    }

    // A `SpliceInst` evaluates the instruction that its operand names.
    if (auto splice = inst.TryAs<SpliceInst>()) {
      CARBON_RETURN_IF_ERROR(
          VerifyOperand(inst_id, splice->inst_id, block_index));
      RecordEvaluated(inst_id);
      if (InstId spliced_id = GetSplicedInstId(*splice);
          spliced_id.has_value()) {
        worklist.push_back({.inst_id = spliced_id});
      }
      continue;
    }

    CARBON_RETURN_IF_ERROR(
        VerifyArg(inst_id, inst.arg0_and_kind(), block_index));
    CARBON_RETURN_IF_ERROR(
        VerifyArg(inst_id, inst.arg1_and_kind(), block_index));
    RecordEvaluated(inst_id);
  }
  return Success();
}

auto DominanceVerifier::VerifyArg(InstId user_id, IdAndKind arg,
                                  BlockIndex block_index) -> ErrorOr<Success> {
  CARBON_KIND_SWITCH(arg) {
    // These operand kinds name the value produced by another instruction, so
    // that instruction's evaluation must dominate this use.
    case CARBON_KIND(InstId inst_id): {
      return VerifyOperand(user_id, inst_id, block_index);
    }
    case CARBON_KIND(DestInstId inst_id): {
      return VerifyOperand(user_id, inst_id, block_index);
    }
    // A `TypeInstId` always names a constant of type `type`, so this check
    // should always pass, but checking it means we notice if that stops being
    // true.
    case CARBON_KIND(TypeInstId inst_id): {
      return VerifyOperand(user_id, inst_id, block_index);
    }
    case CARBON_KIND(InstBlockId block_id): {
      if (block_id.has_value()) {
        for (InstId operand_id : file_.inst_blocks().Get(block_id)) {
          CARBON_RETURN_IF_ERROR(
              VerifyOperand(user_id, operand_id, block_index));
        }
      }
      return Success();
    }
    default: {
      // Every other operand kind either doesn't name an instruction at all, or
      // names one in a way that doesn't require dominance:
      //
      // -   `MetaInstId` and `MetaInstBlockId` name the identity of an
      //     instruction rather than its value.
      // -   `AbsoluteInstId` and `AbsoluteInstBlockId` name instructions that
      //     are typically in a different entity.
      // -   `LabelId` names another block in this function's control flow.
      // -   `DeclInstBlockId` names a declaration rather than a computation.
      return Success();
    }
  }
}

auto DominanceVerifier::VerifyOperand(InstId user_id, InstId operand_id,
                                      BlockIndex block_index)
    -> ErrorOr<Success> {
  if (!operand_id.has_value() || operand_id == ErrorInst::InstId) {
    return Success();
  }
  // A constant isn't evaluated in the function body, so can be used anywhere.
  //
  // TODO: Use `GetConstantValueInSpecific` here, so that an instruction that is
  // only constant in this specific is also exempt. That currently crashes,
  // because a function body can name an instruction whose constant value is
  // attached to an enclosing generic rather than to this function's generic,
  // which `GetConstantInSpecific` rejects. Lowering should hit the same crash;
  // see `FunctionContext::LowerInst`.
  if (file_.constant_values().Get(operand_id).is_constant()) {
    return Success();
  }
  if (evaluated_.Contains(operand_id)) {
    return Success();
  }
  // TODO: Remove this allowlist; see `CollectDeclInsts`.
  if (decl_insts_.Contains(operand_id)) {
    return Success();
  }
  Inst operand = file_.insts().Get(operand_id);

  // TODO: A non-constant import isn't evaluated in the importing file at all,
  // so this is a real violation: `__global_init` can contain a `name_ref` to an
  // `import_ref` for a non-constant imported variable. Remove this exemption
  // and diagnose such uses once imported variables have a value model.
  if (operand.Is<AnyImportRef>()) {
    return Success();
  }

  return ErrorBuilder()
         << "Operand " << operand_id << " (" << operand.kind().ir_name()
         << ") used by instruction " << user_id << " ("
         << file_.insts().Get(user_id).kind().ir_name() << ") in block "
         << body_blocks_[block_index.index] << " of function "
         << function_.name_id
         << " is not dominated by any evaluation and is not constant";
}

auto DominanceVerifier::RecordEvaluated(InstId inst_id) -> void {
  if (!inst_id.has_value()) {
    return;
  }
  // An instruction can be evaluated more than once, for example in two blocks
  // that don't dominate each other. Only the first evaluation in scope needs to
  // be undone.
  if (evaluated_.Insert(inst_id)) {
    evaluated_order_.push_back(inst_id);
  }
}

auto DominanceVerifier::RecordEvaluatedBlock(InstBlockId block_id) -> void {
  if (!block_id.has_value()) {
    return;
  }
  for (InstId inst_id : file_.inst_blocks().Get(block_id)) {
    RecordEvaluated(inst_id);
  }
}

auto DominanceVerifier::GetSplicedInstId(SpliceInst splice) const -> InstId {
  ConstantId const_id =
      GetConstantValueInSpecific(file_, specific_id_, splice.inst_id);
  if (!const_id.is_constant()) {
    return InstId::None;
  }
  InstId const_inst_id = file_.constant_values().GetInstIdIfValid(const_id);
  if (!const_inst_id.has_value()) {
    return InstId::None;
  }
  if (auto inst_value = file_.insts().TryGetAs<InstValue>(const_inst_id)) {
    return inst_value->inst_id;
  }
  return InstId::None;
}

}  // namespace

auto VerifyDominance(const File& file) -> ErrorOr<Success> {
  // Invariants don't necessarily hold for invalid IR.
  if (file.has_errors()) {
    return Success();
  }

  InstSet decl_insts(file);
  CollectDeclInsts(file, decl_insts);

  SpecificsByGeneric specifics = CollectSpecifics(file);

  // Shared by the verification of every function and specific, because creating
  // one costs a bit per instruction in the file. Each `Verify` call leaves it
  // empty for the next one.
  InstSet evaluated(file);

  for (const Function& function : file.functions().values()) {
    if (function.body_block_ids.empty()) {
      continue;
    }

    CARBON_ASSIGN_OR_RETURN(DominatorTree dom_tree,
                            DominatorTreeBuilder(file, function).Build());

    // Verify the body in the general, unspecialized context.
    CARBON_RETURN_IF_ERROR(DominanceVerifier(file, decl_insts, function,
                                             dom_tree, evaluated,
                                             SpecificId::None)
                               .Verify());

    // For a generic function, also verify the body as it will be evaluated in
    // each of its specifics, in which spliced instructions can be resolved.
    for (SpecificId specific_id : specifics.Get(function.generic_id)) {
      CARBON_RETURN_IF_ERROR(DominanceVerifier(file, decl_insts, function,
                                               dom_tree, evaluated, specific_id)
                                 .Verify());
    }
  }

  return Success();
}

}  // namespace Carbon::SemIR
