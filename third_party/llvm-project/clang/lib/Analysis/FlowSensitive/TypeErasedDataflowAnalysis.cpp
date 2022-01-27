//===- TypeErasedDataflowAnalysis.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines type-erased base types and functions for building dataflow
//  analyses that run over Control-Flow Graphs (CFGs).
//
//===----------------------------------------------------------------------===//

#include <utility>
#include <vector>

#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowWorklist.h"
#include "clang/Analysis/FlowSensitive/Transfer.h"
#include "clang/Analysis/FlowSensitive/TypeErasedDataflowAnalysis.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace dataflow {

/// Computes the input state for a given basic block by joining the output
/// states of its predecessors.
///
/// Requirements:
///
///   All predecessors of `Block` except those with loop back edges must have
///   already been transferred. States in `BlockStates` that are set to
///   `llvm::None` represent basic blocks that are not evaluated yet.
static TypeErasedDataflowAnalysisState computeBlockInputState(
    const ControlFlowContext &CFCtx,
    std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>> &BlockStates,
    const CFGBlock &Block, const Environment &InitEnv,
    TypeErasedDataflowAnalysis &Analysis) {
  llvm::DenseSet<const CFGBlock *> Preds;
  Preds.insert(Block.pred_begin(), Block.pred_end());
  if (Block.getTerminator().isTemporaryDtorsBranch()) {
    // This handles a special case where the code that produced the CFG includes
    // a conditional operator with a branch that constructs a temporary and
    // calls a destructor annotated as noreturn. The CFG models this as follows:
    //
    // B1 (contains the condition of the conditional operator) - succs: B2, B3
    // B2 (contains code that does not call a noreturn destructor) - succs: B4
    // B3 (contains code that calls a noreturn destructor) - succs: B4
    // B4 (has temporary destructor terminator) - succs: B5, B6
    // B5 (noreturn block that is associated with the noreturn destructor call)
    // B6 (contains code that follows the conditional operator statement)
    //
    // The first successor (B5 above) of a basic block with a temporary
    // destructor terminator (B4 above) is the block that evaluates the
    // destructor. If that block has a noreturn element then the predecessor
    // block that constructed the temporary object (B3 above) is effectively a
    // noreturn block and its state should not be used as input for the state
    // of the block that has a temporary destructor terminator (B4 above). This
    // holds regardless of which branch of the ternary operator calls the
    // noreturn destructor. However, it doesn't cases where a nested ternary
    // operator includes a branch that contains a noreturn destructor call.
    //
    // See `NoreturnDestructorTest` for concrete examples.
    if (Block.succ_begin()->getReachableBlock()->hasNoReturnElement()) {
      auto StmtBlock = CFCtx.getStmtToBlock().find(Block.getTerminatorStmt());
      assert(StmtBlock != CFCtx.getStmtToBlock().end());
      Preds.erase(StmtBlock->getSecond());
    }
  }

  llvm::Optional<TypeErasedDataflowAnalysisState> MaybeState;
  for (const CFGBlock *Pred : Preds) {
    // Skip if the `Block` is unreachable or control flow cannot get past it.
    if (!Pred || Pred->hasNoReturnElement())
      continue;

    // Skip if `Pred` was not evaluated yet. This could happen if `Pred` has a
    // loop back edge to `Block`.
    const llvm::Optional<TypeErasedDataflowAnalysisState> &MaybePredState =
        BlockStates[Pred->getBlockID()];
    if (!MaybePredState.hasValue())
      continue;

    const TypeErasedDataflowAnalysisState &PredState =
        MaybePredState.getValue();
    if (MaybeState.hasValue()) {
      Analysis.joinTypeErased(MaybeState->Lattice, PredState.Lattice);
      MaybeState->Env.join(PredState.Env);
    } else {
      MaybeState = PredState;
    }
  }
  if (!MaybeState.hasValue()) {
    // FIXME: Consider passing `Block` to `Analysis.typeErasedInitialElement()`
    // to enable building analyses like computation of dominators that
    // initialize the state of each basic block differently.
    MaybeState.emplace(Analysis.typeErasedInitialElement(), InitEnv);
  }
  return *MaybeState;
}

TypeErasedDataflowAnalysisState transferBlock(
    const ControlFlowContext &CFCtx,
    std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>> &BlockStates,
    const CFGBlock &Block, const Environment &InitEnv,
    TypeErasedDataflowAnalysis &Analysis,
    std::function<void(const CFGStmt &,
                       const TypeErasedDataflowAnalysisState &)>
        HandleTransferredStmt) {
  TypeErasedDataflowAnalysisState State =
      computeBlockInputState(CFCtx, BlockStates, Block, InitEnv, Analysis);
  for (const CFGElement &Element : Block) {
    // FIXME: Evaluate other kinds of `CFGElement`.
    const llvm::Optional<CFGStmt> CfgStmt = Element.getAs<CFGStmt>();
    if (!CfgStmt.hasValue())
      continue;

    const Stmt *S = CfgStmt.getValue().getStmt();
    assert(S != nullptr);

    transfer(*S, State.Env);
    Analysis.transferTypeErased(S, State.Lattice, State.Env);

    if (HandleTransferredStmt != nullptr)
      HandleTransferredStmt(CfgStmt.getValue(), State);
  }
  return State;
}

std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>>
runTypeErasedDataflowAnalysis(const ControlFlowContext &CFCtx,
                              TypeErasedDataflowAnalysis &Analysis,
                              const Environment &InitEnv) {
  // FIXME: Consider enforcing that `Cfg` meets the requirements that
  // are specified in the header. This could be done by remembering
  // what options were used to build `Cfg` and asserting on them here.

  PostOrderCFGView POV(&CFCtx.getCFG());
  ForwardDataflowWorklist Worklist(CFCtx.getCFG(), &POV);

  std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>> BlockStates;
  BlockStates.resize(CFCtx.getCFG().size(), llvm::None);

  // The entry basic block doesn't contain statements so it can be skipped.
  const CFGBlock &Entry = CFCtx.getCFG().getEntry();
  BlockStates[Entry.getBlockID()] = {Analysis.typeErasedInitialElement(),
                                     InitEnv};
  Worklist.enqueueSuccessors(&Entry);

  // Bugs in lattices and transfer functions can prevent the analysis from
  // converging. To limit the damage (infinite loops) that these bugs can cause,
  // limit the number of iterations.
  // FIXME: Consider making the maximum number of iterations configurable.
  // FIXME: Set up statistics (see llvm/ADT/Statistic.h) to count average number
  // of iterations, number of functions that time out, etc.
  unsigned Iterations = 0;
  static constexpr unsigned MaxIterations = 1 << 16;
  while (const CFGBlock *Block = Worklist.dequeue()) {
    if (++Iterations > MaxIterations) {
      llvm::errs() << "Maximum number of iterations reached, giving up.\n";
      break;
    }

    const llvm::Optional<TypeErasedDataflowAnalysisState> &OldBlockState =
        BlockStates[Block->getBlockID()];
    TypeErasedDataflowAnalysisState NewBlockState =
        transferBlock(CFCtx, BlockStates, *Block, InitEnv, Analysis);

    if (OldBlockState.hasValue() &&
        Analysis.isEqualTypeErased(OldBlockState.getValue().Lattice,
                                   NewBlockState.Lattice) &&
        OldBlockState->Env == NewBlockState.Env) {
      // The state of `Block` didn't change after transfer so there's no need to
      // revisit its successors.
      continue;
    }

    BlockStates[Block->getBlockID()] = std::move(NewBlockState);

    // Do not add unreachable successor blocks to `Worklist`.
    if (Block->hasNoReturnElement())
      continue;

    Worklist.enqueueSuccessors(Block);
  }
  // FIXME: Consider evaluating unreachable basic blocks (those that have a
  // state set to `llvm::None` at this point) to also analyze dead code.

  return BlockStates;
}

} // namespace dataflow
} // namespace clang
