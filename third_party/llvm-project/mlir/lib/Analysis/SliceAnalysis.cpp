//===- UseDefAnalysis.cpp - Analysis for Transitive UseDef chains ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Analysis functions specific to slicing in Function.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

///
/// Implements Analysis functions specific to slicing in Function.
///

using namespace mlir;

static void getForwardSliceImpl(Operation *op,
                                SetVector<Operation *> *forwardSlice,
                                TransitiveFilter filter) {
  if (!op)
    return;

  // Evaluate whether we should keep this use.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive forwardSlice in the current scope.
  if (filter && !filter(op))
    return;

  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &blockOp : block)
        if (forwardSlice->count(&blockOp) == 0)
          getForwardSliceImpl(&blockOp, forwardSlice, filter);
  for (Value result : op->getResults()) {
    for (Operation *userOp : result.getUsers())
      if (forwardSlice->count(userOp) == 0)
        getForwardSliceImpl(userOp, forwardSlice, filter);
  }

  forwardSlice->insert(op);
}

void mlir::getForwardSlice(Operation *op, SetVector<Operation *> *forwardSlice,
                           TransitiveFilter filter) {
  getForwardSliceImpl(op, forwardSlice, filter);
  // Don't insert the top level operation, we just queried on it and don't
  // want it in the results.
  forwardSlice->remove(op);

  // Reverse to get back the actual topological order.
  // std::reverse does not work out of the box on SetVector and I want an
  // in-place swap based thing (the real std::reverse, not the LLVM adapter).
  std::vector<Operation *> v(forwardSlice->takeVector());
  forwardSlice->insert(v.rbegin(), v.rend());
}

void mlir::getForwardSlice(Value root, SetVector<Operation *> *forwardSlice,
                           TransitiveFilter filter) {
  for (Operation *user : root.getUsers())
    getForwardSliceImpl(user, forwardSlice, filter);

  // Reverse to get back the actual topological order.
  // std::reverse does not work out of the box on SetVector and I want an
  // in-place swap based thing (the real std::reverse, not the LLVM adapter).
  std::vector<Operation *> v(forwardSlice->takeVector());
  forwardSlice->insert(v.rbegin(), v.rend());
}

static void getBackwardSliceImpl(Operation *op,
                                 SetVector<Operation *> *backwardSlice,
                                 TransitiveFilter filter) {
  if (!op || op->hasTrait<OpTrait::IsIsolatedFromAbove>())
    return;

  // Evaluate whether we should keep this def.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive backwardSlice in the current scope.
  if (filter && !filter(op))
    return;

  for (const auto &en : llvm::enumerate(op->getOperands())) {
    auto operand = en.value();
    if (auto *definingOp = operand.getDefiningOp()) {
      if (backwardSlice->count(definingOp) == 0)
        getBackwardSliceImpl(definingOp, backwardSlice, filter);
    } else if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
      Block *block = blockArg.getOwner();
      Operation *parentOp = block->getParentOp();
      // TODO: determine whether we want to recurse backward into the other
      // blocks of parentOp, which are not technically backward unless they flow
      // into us. For now, just bail.
      assert(parentOp->getNumRegions() == 1 &&
             parentOp->getRegion(0).getBlocks().size() == 1);
      if (backwardSlice->count(parentOp) == 0)
        getBackwardSliceImpl(parentOp, backwardSlice, filter);
    } else {
      llvm_unreachable("No definingOp and not a block argument.");
    }
  }

  backwardSlice->insert(op);
}

void mlir::getBackwardSlice(Operation *op,
                            SetVector<Operation *> *backwardSlice,
                            TransitiveFilter filter) {
  getBackwardSliceImpl(op, backwardSlice, filter);

  // Don't insert the top level operation, we just queried on it and don't
  // want it in the results.
  backwardSlice->remove(op);
}

void mlir::getBackwardSlice(Value root, SetVector<Operation *> *backwardSlice,
                            TransitiveFilter filter) {
  if (Operation *definingOp = root.getDefiningOp()) {
    getBackwardSlice(definingOp, backwardSlice, filter);
    return;
  }
  Operation *bbAargOwner = root.cast<BlockArgument>().getOwner()->getParentOp();
  getBackwardSlice(bbAargOwner, backwardSlice, filter);
}

SetVector<Operation *> mlir::getSlice(Operation *op,
                                      TransitiveFilter backwardFilter,
                                      TransitiveFilter forwardFilter) {
  SetVector<Operation *> slice;
  slice.insert(op);

  unsigned currentIndex = 0;
  SetVector<Operation *> backwardSlice;
  SetVector<Operation *> forwardSlice;
  while (currentIndex != slice.size()) {
    auto *currentOp = (slice)[currentIndex];
    // Compute and insert the backwardSlice starting from currentOp.
    backwardSlice.clear();
    getBackwardSlice(currentOp, &backwardSlice, backwardFilter);
    slice.insert(backwardSlice.begin(), backwardSlice.end());

    // Compute and insert the forwardSlice starting from currentOp.
    forwardSlice.clear();
    getForwardSlice(currentOp, &forwardSlice, forwardFilter);
    slice.insert(forwardSlice.begin(), forwardSlice.end());
    ++currentIndex;
  }
  return topologicalSort(slice);
}

namespace {
/// DFS post-order implementation that maintains a global count to work across
/// multiple invocations, to help implement topological sort on multi-root DAGs.
/// We traverse all operations but only record the ones that appear in
/// `toSort` for the final result.
struct DFSState {
  DFSState(const SetVector<Operation *> &set)
      : toSort(set), topologicalCounts(), seen() {}
  const SetVector<Operation *> &toSort;
  SmallVector<Operation *, 16> topologicalCounts;
  DenseSet<Operation *> seen;
};
} // namespace

static void dfsPostorder(Operation *root, DFSState *state) {
  SmallVector<Operation *> queue(1, root);
  std::vector<Operation *> ops;
  while (!queue.empty()) {
    Operation *current = queue.pop_back_val();
    ops.push_back(current);
    for (Value result : current->getResults()) {
      for (Operation *op : result.getUsers())
        queue.push_back(op);
    }
    for (Region &region : current->getRegions()) {
      for (Operation &op : region.getOps())
        queue.push_back(&op);
    }
  }

  for (Operation *op : llvm::reverse(ops)) {
    if (state->seen.insert(op).second && state->toSort.count(op) > 0)
      state->topologicalCounts.push_back(op);
  }
}

SetVector<Operation *>
mlir::topologicalSort(const SetVector<Operation *> &toSort) {
  if (toSort.empty()) {
    return toSort;
  }

  // Run from each root with global count and `seen` set.
  DFSState state(toSort);
  for (auto *s : toSort) {
    assert(toSort.count(s) == 1 && "NYI: multi-sets not supported");
    dfsPostorder(s, &state);
  }

  // Reorder and return.
  SetVector<Operation *> res;
  for (auto it = state.topologicalCounts.rbegin(),
            eit = state.topologicalCounts.rend();
       it != eit; ++it) {
    res.insert(*it);
  }
  return res;
}

/// Returns true if `value` (transitively) depends on iteration-carried values
/// of the given `ancestorOp`.
static bool dependsOnCarriedVals(Value value,
                                 ArrayRef<BlockArgument> iterCarriedArgs,
                                 Operation *ancestorOp) {
  // Compute the backward slice of the value.
  SetVector<Operation *> slice;
  getBackwardSlice(value, &slice,
                   [&](Operation *op) { return !ancestorOp->isAncestor(op); });

  // Check that none of the operands of the operations in the backward slice are
  // loop iteration arguments, and neither is the value itself.
  SmallPtrSet<Value, 8> iterCarriedValSet(iterCarriedArgs.begin(),
                                          iterCarriedArgs.end());
  if (iterCarriedValSet.contains(value))
    return true;

  for (Operation *op : slice)
    for (Value operand : op->getOperands())
      if (iterCarriedValSet.contains(operand))
        return true;

  return false;
}

/// Utility to match a generic reduction given a list of iteration-carried
/// arguments, `iterCarriedArgs` and the position of the potential reduction
/// argument within the list, `redPos`. If a reduction is matched, returns the
/// reduced value and the topologically-sorted list of combiner operations
/// involved in the reduction. Otherwise, returns a null value.
///
/// The matching algorithm relies on the following invariants, which are subject
/// to change:
///  1. The first combiner operation must be a binary operation with the
///     iteration-carried value and the reduced value as operands.
///  2. The iteration-carried value and combiner operations must be side
///     effect-free, have single result and a single use.
///  3. Combiner operations must be immediately nested in the region op
///     performing the reduction.
///  4. Reduction def-use chain must end in a terminator op that yields the
///     next iteration/output values in the same order as the iteration-carried
///     values in `iterCarriedArgs`.
///  5. `iterCarriedArgs` must contain all the iteration-carried/output values
///     of the region op performing the reduction.
///
/// This utility is generic enough to detect reductions involving multiple
/// combiner operations (disabled for now) across multiple dialects, including
/// Linalg, Affine and SCF. For the sake of genericity, it does not return
/// specific enum values for the combiner operations since its goal is also
/// matching reductions without pre-defined semantics in core MLIR. It's up to
/// each client to make sense out of the list of combiner operations. It's also
/// up to each client to check for additional invariants on the expected
/// reductions not covered by this generic matching.
Value mlir::matchReduction(ArrayRef<BlockArgument> iterCarriedArgs,
                           unsigned redPos,
                           SmallVectorImpl<Operation *> &combinerOps) {
  assert(redPos < iterCarriedArgs.size() && "'redPos' is out of bounds");

  BlockArgument redCarriedVal = iterCarriedArgs[redPos];
  if (!redCarriedVal.hasOneUse())
    return nullptr;

  // For now, the first combiner op must be a binary op.
  Operation *combinerOp = *redCarriedVal.getUsers().begin();
  if (combinerOp->getNumOperands() != 2)
    return nullptr;
  Value reducedVal = combinerOp->getOperand(0) == redCarriedVal
                         ? combinerOp->getOperand(1)
                         : combinerOp->getOperand(0);

  Operation *redRegionOp =
      iterCarriedArgs.front().getOwner()->getParent()->getParentOp();
  if (dependsOnCarriedVals(reducedVal, iterCarriedArgs, redRegionOp))
    return nullptr;

  // Traverse the def-use chain starting from the first combiner op until a
  // terminator is found. Gather all the combiner ops along the way in
  // topological order.
  while (!combinerOp->mightHaveTrait<OpTrait::IsTerminator>()) {
    if (!MemoryEffectOpInterface::hasNoEffect(combinerOp) ||
        combinerOp->getNumResults() != 1 || !combinerOp->hasOneUse() ||
        combinerOp->getParentOp() != redRegionOp)
      return nullptr;

    combinerOps.push_back(combinerOp);
    combinerOp = *combinerOp->getUsers().begin();
  }

  // Limit matching to single combiner op until we can properly test reductions
  // involving multiple combiners.
  if (combinerOps.size() != 1)
    return nullptr;

  // Check that the yielded value is in the same position as in
  // `iterCarriedArgs`.
  Operation *terminatorOp = combinerOp;
  if (terminatorOp->getOperand(redPos) != combinerOps.back()->getResults()[0])
    return nullptr;

  return reducedVal;
}
