// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/dominance.h"

#include <algorithm>
#include <utility>

#include "common/check.h"
#include "common/error.h"
#include "common/map.h"
#include "common/set.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/index_base.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
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

// The maximum depth to which we look through spliced instructions. This is a
// safeguard against malformed IR in which spliced instructions form a cycle;
// well-formed IR nests far more shallowly than this.
constexpr int MaxSpliceDepth = 50;

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
  if (auto branch = inst.TryAs<Branch>()) {
    return branch->target_id;
  }
  if (auto branch_if = inst.TryAs<BranchIf>()) {
    return branch_if->target_id;
  }
  if (auto branch_with_arg = inst.TryAs<BranchWithArg>()) {
    return branch_with_arg->target_id;
  }
  return InstBlockId::None;
}

// Collects the instructions that form declarations, either at file scope or
// within an entity, into `decl_insts`.
//
// TODO: These instructions are referenced from within function bodies even
// though they are not evaluated there: file-scope variables, class / interface
// / impl members, and imports all leak into function bodies as direct
// instruction references. They are allowlisted so that the dominance check
// doesn't reject the existing test corpus. Once global initialization semantics
// and the variable reference model are settled, remove this allowlist and
// diagnose such uses instead.
auto CollectDeclInsts(const File& file, Set<InstId>& decl_insts) -> void {
  auto add_block = [&](InstBlockId block_id) {
    if (block_id.has_value()) {
      for (InstId inst_id : file.inst_blocks().Get(block_id)) {
        decl_insts.Insert(inst_id);
      }
    }
  };
  auto add_decl = [&](InstId decl_id) {
    if (!decl_id.has_value()) {
      return;
    }
    Inst inst = file.insts().Get(decl_id);
    if (auto decl = inst.TryAs<ClassDecl>()) {
      add_block(decl->decl_block_id);
    } else if (auto decl = inst.TryAs<FunctionDecl>()) {
      add_block(decl->decl_block_id);
    } else if (auto decl = inst.TryAs<InterfaceDecl>()) {
      add_block(decl->decl_block_id);
    } else if (auto decl = inst.TryAs<ImplDecl>()) {
      add_block(decl->decl_block_id);
    }
  };
  auto add_entity = [&](const EntityWithParamsBase& entity) {
    add_decl(entity.definition_id);
    add_decl(entity.first_owning_decl_id);
    add_decl(entity.non_owning_decl_id);
    add_block(entity.pattern_block_id);
  };

  add_block(file.top_inst_block_id());
  for (const auto& function : file.functions().values()) {
    add_entity(function);
  }
  for (const auto& class_info : file.classes().values()) {
    add_entity(class_info);
    add_block(class_info.body_block_id);
  }
  for (const auto& interface : file.interfaces().values()) {
    add_entity(interface);
    add_block(interface.body_block_without_self_id);
    add_block(interface.body_block_with_self_id);
    add_block(interface.associated_entities_id);
  }
  for (const auto& impl : file.impls().values()) {
    add_entity(impl);
    add_block(impl.body_block_id);
    add_block(impl.witness_block_id);
  }
}

// Verifies that every operand of every instruction in one function body is
// dominated by an evaluation of that operand.
//
// This walks the dominator tree of the function's control flow graph, tracking
// the set of instructions whose evaluations dominate the point currently being
// checked. An instruction joins that set when its evaluation is reached, and
// leaves it again when the walk leaves the blocks that the evaluation
// dominates.
class DominanceVerifier {
 public:
  explicit DominanceVerifier(const File& file, const Set<InstId>& decl_insts,
                             const Function& function, SpecificId specific_id)
      : file_(file),
        decl_insts_(decl_insts),
        function_(function),
        specific_id_(specific_id),
        body_blocks_(function.body_block_ids) {}

  auto Verify() -> ErrorOr<Success>;

 private:
  // Builds `successors_` and `predecessors_` from the branches in each block.
  auto BuildControlFlowGraph() -> ErrorOr<Success>;

  // Builds `dom_children_`, and diagnoses blocks that are unreachable from the
  // entry block.
  auto BuildDominatorTree() -> ErrorOr<Success>;

  // Appends the blocks reachable from the entry block to `post_order`, in
  // post-order, and marks each of them `visited`.
  auto BuildPostOrder(llvm::SmallVectorImpl<bool>& visited,
                      llvm::SmallVectorImpl<BlockIndex>& post_order) -> void;

  // Verifies every block, walking the dominator tree from the entry block.
  auto VerifyBlocks() -> ErrorOr<Success>;

  // Verifies the operands of `inst_id`, then records `inst_id`, along with any
  // instructions it splices into the enclosing block, as evaluated.
  auto VerifyAndRecordInst(InstId inst_id, BlockIndex block_index, int depth)
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

  // Returns the constant value of `inst_id` in the specific being verified.
  auto GetConstantValue(InstId inst_id) const -> ConstantId;

  // Returns the instruction that `splice` splices in, or `InstId::None` if that
  // can't be determined.
  auto GetSplicedInstId(SpliceInst splice) const -> InstId;

  auto num_blocks() const -> int { return body_blocks_.size(); }

  const File& file_;
  const Set<InstId>& decl_insts_;
  const Function& function_;
  SpecificId specific_id_;
  llvm::ArrayRef<InstBlockId> body_blocks_;

  // The index of each block in `body_blocks_`.
  Map<InstBlockId, BlockIndex> block_indexes_;

  // The control flow graph and dominator tree, indexed by `BlockIndex`.
  llvm::SmallVector<llvm::SmallVector<BlockIndex, 2>> successors_;
  llvm::SmallVector<llvm::SmallVector<BlockIndex, 2>> predecessors_;
  llvm::SmallVector<llvm::SmallVector<BlockIndex, 2>> dom_children_;

  // The instructions whose evaluations dominate the point currently being
  // verified, and the order in which they were added, so that they can be
  // removed again when leaving a block.
  Set<InstId> evaluated_;
  llvm::SmallVector<InstId> evaluated_order_;
};

auto DominanceVerifier::Verify() -> ErrorOr<Success> {
  CARBON_CHECK(!body_blocks_.empty());

  CARBON_RETURN_IF_ERROR(BuildControlFlowGraph());
  CARBON_RETURN_IF_ERROR(BuildDominatorTree());

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

  return VerifyBlocks();
}

auto DominanceVerifier::BuildControlFlowGraph() -> ErrorOr<Success> {
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
      if (!llvm::is_contained(successors_[from.index], *to)) {
        successors_[from.index].push_back(*to);
        predecessors_[to->index].push_back(from);
      }
    }
  }
  return Success();
}

auto DominanceVerifier::BuildPostOrder(
    llvm::SmallVectorImpl<bool>& visited,
    llvm::SmallVectorImpl<BlockIndex>& post_order) -> void {
  // The blocks whose successors are still being visited, each paired with the
  // number of its successors that have been visited so far. A block is appended
  // to `post_order` once all of its successors have been visited.
  llvm::SmallVector<std::pair<BlockIndex, int>> stack;
  visited[EntryBlockIndex.index] = true;
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
    if (!visited[successor.index]) {
      visited[successor.index] = true;
      stack.push_back({successor, 0});
    }
  }
}

auto DominanceVerifier::BuildDominatorTree() -> ErrorOr<Success> {
  // Order the blocks so that, apart from loop back edges, every block precedes
  // its successors.
  llvm::SmallVector<bool> visited(num_blocks(), false);
  llvm::SmallVector<BlockIndex> reverse_post_order;
  reverse_post_order.reserve(num_blocks());
  BuildPostOrder(visited, reverse_post_order);
  std::reverse(reverse_post_order.begin(), reverse_post_order.end());

  for (int i = 0; i != num_blocks(); ++i) {
    if (!visited[i]) {
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

  dom_children_.resize(num_blocks());
  for (BlockIndex block_index : llvm::drop_begin(reverse_post_order)) {
    // Every block other than the entry block is reachable, and so is dominated
    // by the predecessor it's reached through.
    CARBON_CHECK(idom[block_index.index].has_value(), "No dominator for {0}",
                 body_blocks_[block_index.index]);
    dom_children_[idom[block_index.index].index].push_back(block_index);
  }
  return Success();
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
      CARBON_RETURN_IF_ERROR(
          VerifyAndRecordInst(inst_id, block_index, /*depth=*/0));
    }
    for (BlockIndex child : dom_children_[block_index.index]) {
      worklist.push_back(WalkStep::EnterBlock(child));
    }
  }
  return Success();
}

auto DominanceVerifier::VerifyAndRecordInst(InstId inst_id,
                                            BlockIndex block_index, int depth)
    -> ErrorOr<Success> {
  if (depth > MaxSpliceDepth) {
    return ErrorBuilder() << "Spliced instructions are nested more than "
                          << MaxSpliceDepth << " deep at instruction "
                          << inst_id << " in function " << function_.name_id;
  }

  Inst inst = file_.insts().Get(inst_id);

  // A `SpliceBlock` evaluates the instructions in its block, and then produces
  // the value of one of them.
  if (auto splice_block = inst.TryAs<SpliceBlock>()) {
    if (splice_block->block_id.has_value()) {
      for (InstId spliced_id :
           file_.inst_blocks().Get(splice_block->block_id)) {
        CARBON_RETURN_IF_ERROR(
            VerifyAndRecordInst(spliced_id, block_index, depth + 1));
      }
    }
    CARBON_RETURN_IF_ERROR(
        VerifyOperand(inst_id, splice_block->result_id, block_index));
    RecordEvaluated(inst_id);
    return Success();
  }

  // A `SpliceInst` evaluates the instruction that its operand names.
  if (auto splice = inst.TryAs<SpliceInst>()) {
    CARBON_RETURN_IF_ERROR(
        VerifyOperand(inst_id, splice->inst_id, block_index));
    RecordEvaluated(inst_id);
    if (InstId spliced_id = GetSplicedInstId(*splice); spliced_id.has_value()) {
      CARBON_RETURN_IF_ERROR(
          VerifyAndRecordInst(spliced_id, block_index, depth + 1));
    }
    return Success();
  }

  CARBON_RETURN_IF_ERROR(VerifyArg(inst_id, inst.arg0_and_kind(), block_index));
  CARBON_RETURN_IF_ERROR(VerifyArg(inst_id, inst.arg1_and_kind(), block_index));
  RecordEvaluated(inst_id);
  return Success();
}

auto DominanceVerifier::VerifyArg(InstId user_id, IdAndKind arg,
                                  BlockIndex block_index) -> ErrorOr<Success> {
  // These operand kinds name the value produced by another instruction, so that
  // instruction's evaluation must dominate this use.
  if (arg.kind() == IdKind::For<InstId>) {
    return VerifyOperand(user_id, arg.As<InstId>(), block_index);
  }
  if (arg.kind() == IdKind::For<DestInstId>) {
    return VerifyOperand(user_id, arg.As<DestInstId>(), block_index);
  }
  // A `TypeInstId` always names a constant of type `type`, so this check should
  // always pass, but checking it means we notice if that stops being true.
  if (arg.kind() == IdKind::For<TypeInstId>) {
    return VerifyOperand(user_id, arg.As<TypeInstId>(), block_index);
  }
  if (arg.kind() == IdKind::For<InstBlockId>) {
    InstBlockId block_id = arg.As<InstBlockId>();
    if (block_id.has_value()) {
      for (InstId operand_id : file_.inst_blocks().Get(block_id)) {
        CARBON_RETURN_IF_ERROR(VerifyOperand(user_id, operand_id, block_index));
      }
    }
    return Success();
  }

  // Every other operand kind either doesn't name an instruction at all, or
  // names one in a way that doesn't require dominance:
  //
  // -   `MetaInstId` and `MetaInstBlockId` name the identity of an instruction
  //     rather than its value.
  // -   `AbsoluteInstId` and `AbsoluteInstBlockId` name instructions that are
  //     typically in a different entity.
  // -   `LabelId` names another block in this function's control flow.
  // -   `DeclInstBlockId` names a declaration rather than a computation.
  return Success();
}

auto DominanceVerifier::VerifyOperand(InstId user_id, InstId operand_id,
                                      BlockIndex block_index)
    -> ErrorOr<Success> {
  if (!operand_id.has_value() || operand_id == ErrorInst::InstId) {
    return Success();
  }
  // A constant isn't evaluated in the function body, so can be used anywhere.
  if (GetConstantValue(operand_id).is_constant()) {
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
  if (operand.Is<ImportRefLoaded>() || operand.Is<ImportRefUnloaded>()) {
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
  if (evaluated_.Insert(inst_id).is_inserted()) {
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

auto DominanceVerifier::GetConstantValue(InstId inst_id) const -> ConstantId {
  ConstantId const_id = file_.constant_values().Get(inst_id);
  if (!const_id.has_value()) {
    return ConstantId::None;
  }
  if (!const_id.is_symbolic()) {
    return const_id;
  }
  // In a specific, a symbolic constant of the corresponding generic has a more
  // specific value that we should use instead.
  if (specific_id_.has_value()) {
    const auto& symbolic =
        file_.constant_values().GetSymbolicConstant(const_id);
    if (symbolic.generic_id.has_value() &&
        symbolic.generic_id == file_.specifics().Get(specific_id_).generic_id) {
      return GetConstantValueInSpecific(file_, specific_id_, inst_id);
    }
  }
  return const_id;
}

auto DominanceVerifier::GetSplicedInstId(SpliceInst splice) const -> InstId {
  ConstantId const_id = GetConstantValue(splice.inst_id);
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

  Set<InstId> decl_insts;
  CollectDeclInsts(file, decl_insts);

  for (const Function& function : file.functions().values()) {
    if (function.body_block_ids.empty()) {
      continue;
    }

    // Verify the body in the general, unspecialized context.
    CARBON_RETURN_IF_ERROR(
        DominanceVerifier(file, decl_insts, function, SpecificId::None)
            .Verify());

    // For a generic function, also verify the body as it will be evaluated in
    // each of its specifics, in which spliced instructions can be resolved.
    if (!function.generic_id.has_value()) {
      continue;
    }
    for (const auto& [specific_id, specific] : file.specifics().enumerate()) {
      if (specific.generic_id == function.generic_id &&
          !specific.IsUnresolved() && !specific.HasError()) {
        CARBON_RETURN_IF_ERROR(
            DominanceVerifier(file, decl_insts, function, specific_id)
                .Verify());
      }
    }
  }

  return Success();
}

}  // namespace Carbon::SemIR
