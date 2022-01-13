//===- PipelineDataTransfer.cpp --- Pass for pipelining data movement ---*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to pipeline data transfers.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-pipeline-data-transfer"

using namespace mlir;

namespace {
struct PipelineDataTransfer
    : public AffinePipelineDataTransferBase<PipelineDataTransfer> {
  void runOnFunction() override;
  void runOnAffineForOp(AffineForOp forOp);

  std::vector<AffineForOp> forOps;
};

} // end anonymous namespace

/// Creates a pass to pipeline explicit movement of data across levels of the
/// memory hierarchy.
std::unique_ptr<OperationPass<FuncOp>> mlir::createPipelineDataTransferPass() {
  return std::make_unique<PipelineDataTransfer>();
}

// Returns the position of the tag memref operand given a DMA operation.
// Temporary utility: will be replaced when DmaStart/DmaFinish abstract op's are
// added.  TODO
static unsigned getTagMemRefPos(Operation &dmaOp) {
  assert((isa<AffineDmaStartOp, AffineDmaWaitOp>(dmaOp)));
  if (auto dmaStartOp = dyn_cast<AffineDmaStartOp>(dmaOp)) {
    return dmaStartOp.getTagMemRefOperandIndex();
  }
  // First operand for a dma finish operation.
  return 0;
}

/// Doubles the buffer of the supplied memref on the specified 'affine.for'
/// operation by adding a leading dimension of size two to the memref.
/// Replaces all uses of the old memref by the new one while indexing the newly
/// added dimension by the loop IV of the specified 'affine.for' operation
/// modulo 2. Returns false if such a replacement cannot be performed.
static bool doubleBuffer(Value oldMemRef, AffineForOp forOp) {
  auto *forBody = forOp.getBody();
  OpBuilder bInner(forBody, forBody->begin());

  // Doubles the shape with a leading dimension extent of 2.
  auto doubleShape = [&](MemRefType oldMemRefType) -> MemRefType {
    // Add the leading dimension in the shape for the double buffer.
    ArrayRef<int64_t> oldShape = oldMemRefType.getShape();
    SmallVector<int64_t, 4> newShape(1 + oldMemRefType.getRank());
    newShape[0] = 2;
    std::copy(oldShape.begin(), oldShape.end(), newShape.begin() + 1);
    return MemRefType::Builder(oldMemRefType)
        .setShape(newShape)
        .setAffineMaps({});
  };

  auto oldMemRefType = oldMemRef.getType().cast<MemRefType>();
  auto newMemRefType = doubleShape(oldMemRefType);

  // The double buffer is allocated right before 'forOp'.
  OpBuilder bOuter(forOp);
  // Put together alloc operands for any dynamic dimensions of the memref.
  SmallVector<Value, 4> allocOperands;
  for (auto dim : llvm::enumerate(oldMemRefType.getShape())) {
    if (dim.value() == ShapedType::kDynamicSize)
      allocOperands.push_back(bOuter.createOrFold<memref::DimOp>(
          forOp.getLoc(), oldMemRef, dim.index()));
  }

  // Create and place the alloc right before the 'affine.for' operation.
  Value newMemRef = bOuter.create<memref::AllocOp>(
      forOp.getLoc(), newMemRefType, allocOperands);

  // Create 'iv mod 2' value to index the leading dimension.
  auto d0 = bInner.getAffineDimExpr(0);
  int64_t step = forOp.getStep();
  auto modTwoMap =
      AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, d0.floorDiv(step) % 2);
  auto ivModTwoOp = bInner.create<AffineApplyOp>(forOp.getLoc(), modTwoMap,
                                                 forOp.getInductionVar());

  // replaceAllMemRefUsesWith will succeed unless the forOp body has
  // non-dereferencing uses of the memref (dealloc's are fine though).
  if (failed(replaceAllMemRefUsesWith(
          oldMemRef, newMemRef,
          /*extraIndices=*/{ivModTwoOp},
          /*indexRemap=*/AffineMap(),
          /*extraOperands=*/{},
          /*symbolOperands=*/{},
          /*domInstFilter=*/&*forOp.getBody()->begin()))) {
    LLVM_DEBUG(
        forOp.emitError("memref replacement for double buffering failed"));
    ivModTwoOp.erase();
    return false;
  }
  // Insert the dealloc op right after the for loop.
  bOuter.setInsertionPointAfter(forOp);
  bOuter.create<memref::DeallocOp>(forOp.getLoc(), newMemRef);

  return true;
}

/// Returns success if the IR is in a valid state.
void PipelineDataTransfer::runOnFunction() {
  // Do a post order walk so that inner loop DMAs are processed first. This is
  // necessary since 'affine.for' operations nested within would otherwise
  // become invalid (erased) when the outer loop is pipelined (the pipelined one
  // gets deleted and replaced by a prologue, a new steady-state loop and an
  // epilogue).
  forOps.clear();
  getFunction().walk([&](AffineForOp forOp) { forOps.push_back(forOp); });
  for (auto forOp : forOps)
    runOnAffineForOp(forOp);
}

// Check if tags of the dma start op and dma wait op match.
static bool checkTagMatch(AffineDmaStartOp startOp, AffineDmaWaitOp waitOp) {
  if (startOp.getTagMemRef() != waitOp.getTagMemRef())
    return false;
  auto startIndices = startOp.getTagIndices();
  auto waitIndices = waitOp.getTagIndices();
  // Both of these have the same number of indices since they correspond to the
  // same tag memref.
  for (auto it = startIndices.begin(), wIt = waitIndices.begin(),
            e = startIndices.end();
       it != e; ++it, ++wIt) {
    // Keep it simple for now, just checking if indices match.
    // TODO: this would in general need to check if there is no
    // intervening write writing to the same tag location, i.e., memory last
    // write/data flow analysis. This is however sufficient/powerful enough for
    // now since the DMA generation pass or the input for it will always have
    // start/wait with matching tags (same SSA operand indices).
    if (*it != *wIt)
      return false;
  }
  return true;
}

// Identify matching DMA start/finish operations to overlap computation with.
static void findMatchingStartFinishInsts(
    AffineForOp forOp,
    SmallVectorImpl<std::pair<Operation *, Operation *>> &startWaitPairs) {

  // Collect outgoing DMA operations - needed to check for dependences below.
  SmallVector<AffineDmaStartOp, 4> outgoingDmaOps;
  for (auto &op : *forOp.getBody()) {
    auto dmaStartOp = dyn_cast<AffineDmaStartOp>(op);
    if (dmaStartOp && dmaStartOp.isSrcMemorySpaceFaster())
      outgoingDmaOps.push_back(dmaStartOp);
  }

  SmallVector<Operation *, 4> dmaStartInsts, dmaFinishInsts;
  for (auto &op : *forOp.getBody()) {
    // Collect DMA finish operations.
    if (isa<AffineDmaWaitOp>(op)) {
      dmaFinishInsts.push_back(&op);
      continue;
    }
    auto dmaStartOp = dyn_cast<AffineDmaStartOp>(op);
    if (!dmaStartOp)
      continue;

    // Only DMAs incoming into higher memory spaces are pipelined for now.
    // TODO: handle outgoing DMA pipelining.
    if (!dmaStartOp.isDestMemorySpaceFaster())
      continue;

    // Check for dependence with outgoing DMAs. Doing this conservatively.
    // TODO: use the dependence analysis to check for
    // dependences between an incoming and outgoing DMA in the same iteration.
    auto it = outgoingDmaOps.begin();
    for (; it != outgoingDmaOps.end(); ++it) {
      if (it->getDstMemRef() == dmaStartOp.getSrcMemRef())
        break;
    }
    if (it != outgoingDmaOps.end())
      continue;

    // We only double buffer if the buffer is not live out of loop.
    auto memref = dmaStartOp.getOperand(dmaStartOp.getFasterMemPos());
    bool escapingUses = false;
    for (auto *user : memref.getUsers()) {
      // We can double buffer regardless of dealloc's outside the loop.
      if (isa<memref::DeallocOp>(user))
        continue;
      if (!forOp.getBody()->findAncestorOpInBlock(*user)) {
        LLVM_DEBUG(llvm::dbgs()
                       << "can't pipeline: buffer is live out of loop\n";);
        escapingUses = true;
        break;
      }
    }
    if (!escapingUses)
      dmaStartInsts.push_back(&op);
  }

  // For each start operation, we look for a matching finish operation.
  for (auto *dmaStartOp : dmaStartInsts) {
    for (auto *dmaFinishOp : dmaFinishInsts) {
      if (checkTagMatch(cast<AffineDmaStartOp>(dmaStartOp),
                        cast<AffineDmaWaitOp>(dmaFinishOp))) {
        startWaitPairs.push_back({dmaStartOp, dmaFinishOp});
        break;
      }
    }
  }
}

/// Overlap DMA transfers with computation in this loop. If successful,
/// 'forOp' is deleted, and a prologue, a new pipelined loop, and epilogue are
/// inserted right before where it was.
void PipelineDataTransfer::runOnAffineForOp(AffineForOp forOp) {
  auto mayBeConstTripCount = getConstantTripCount(forOp);
  if (!mayBeConstTripCount.hasValue()) {
    LLVM_DEBUG(forOp.emitRemark("won't pipeline due to unknown trip count"));
    return;
  }

  SmallVector<std::pair<Operation *, Operation *>, 4> startWaitPairs;
  findMatchingStartFinishInsts(forOp, startWaitPairs);

  if (startWaitPairs.empty()) {
    LLVM_DEBUG(forOp.emitRemark("No dma start/finish pairs\n"));
    return;
  }

  // Double the buffers for the higher memory space memref's.
  // Identify memref's to replace by scanning through all DMA start
  // operations. A DMA start operation has two memref's - the one from the
  // higher level of memory hierarchy is the one to double buffer.
  // TODO: check whether double-buffering is even necessary.
  // TODO: make this work with different layouts: assuming here that
  // the dimension we are adding here for the double buffering is the outermost
  // dimension.
  for (auto &pair : startWaitPairs) {
    auto *dmaStartOp = pair.first;
    Value oldMemRef = dmaStartOp->getOperand(
        cast<AffineDmaStartOp>(dmaStartOp).getFasterMemPos());
    if (!doubleBuffer(oldMemRef, forOp)) {
      // Normally, double buffering should not fail because we already checked
      // that there are no uses outside.
      LLVM_DEBUG(llvm::dbgs()
                     << "double buffering failed for" << dmaStartOp << "\n";);
      // IR still valid and semantically correct.
      return;
    }
    // If the old memref has no more uses, remove its 'dead' alloc if it was
    // alloc'ed. (note: DMA buffers are rarely function live-in; but a 'dim'
    // operation could have been used on it if it was dynamically shaped in
    // order to create the double buffer above.)
    // '-canonicalize' does this in a more general way, but we'll anyway do the
    // simple/common case so that the output / test cases looks clear.
    if (auto *allocOp = oldMemRef.getDefiningOp()) {
      if (oldMemRef.use_empty()) {
        allocOp->erase();
      } else if (oldMemRef.hasOneUse()) {
        if (auto dealloc =
                dyn_cast<memref::DeallocOp>(*oldMemRef.user_begin())) {
          dealloc.erase();
          allocOp->erase();
        }
      }
    }
  }

  // Double the buffers for tag memrefs.
  for (auto &pair : startWaitPairs) {
    auto *dmaFinishOp = pair.second;
    Value oldTagMemRef = dmaFinishOp->getOperand(getTagMemRefPos(*dmaFinishOp));
    if (!doubleBuffer(oldTagMemRef, forOp)) {
      LLVM_DEBUG(llvm::dbgs() << "tag double buffering failed\n";);
      return;
    }
    // If the old tag has no uses or a single dealloc use, remove it.
    // (canonicalization handles more complex cases).
    if (auto *tagAllocOp = oldTagMemRef.getDefiningOp()) {
      if (oldTagMemRef.use_empty()) {
        tagAllocOp->erase();
      } else if (oldTagMemRef.hasOneUse()) {
        if (auto dealloc =
                dyn_cast<memref::DeallocOp>(*oldTagMemRef.user_begin())) {
          dealloc.erase();
          tagAllocOp->erase();
        }
      }
    }
  }

  // Double buffering would have invalidated all the old DMA start/wait insts.
  startWaitPairs.clear();
  findMatchingStartFinishInsts(forOp, startWaitPairs);

  // Store shift for operation for later lookup for AffineApplyOp's.
  DenseMap<Operation *, unsigned> instShiftMap;
  for (auto &pair : startWaitPairs) {
    auto *dmaStartOp = pair.first;
    assert(isa<AffineDmaStartOp>(dmaStartOp));
    instShiftMap[dmaStartOp] = 0;
    // Set shifts for DMA start op's affine operand computation slices to 0.
    SmallVector<AffineApplyOp, 4> sliceOps;
    mlir::createAffineComputationSlice(dmaStartOp, &sliceOps);
    if (!sliceOps.empty()) {
      for (auto sliceOp : sliceOps) {
        instShiftMap[sliceOp.getOperation()] = 0;
      }
    } else {
      // If a slice wasn't created, the reachable affine.apply op's from its
      // operands are the ones that go with it.
      SmallVector<Operation *, 4> affineApplyInsts;
      SmallVector<Value, 4> operands(dmaStartOp->getOperands());
      getReachableAffineApplyOps(operands, affineApplyInsts);
      for (auto *op : affineApplyInsts) {
        instShiftMap[op] = 0;
      }
    }
  }
  // Everything else (including compute ops and dma finish) are shifted by one.
  for (auto &op : forOp.getBody()->without_terminator())
    if (instShiftMap.find(&op) == instShiftMap.end())
      instShiftMap[&op] = 1;

  // Get shifts stored in map.
  SmallVector<uint64_t, 8> shifts(forOp.getBody()->getOperations().size());
  unsigned s = 0;
  for (auto &op : forOp.getBody()->without_terminator()) {
    assert(instShiftMap.find(&op) != instShiftMap.end());
    shifts[s++] = instShiftMap[&op];

    // Tagging operations with shifts for debugging purposes.
    LLVM_DEBUG({
      OpBuilder b(&op);
      op.setAttr("shift", b.getI64IntegerAttr(shifts[s - 1]));
    });
  }

  if (!isOpwiseShiftValid(forOp, shifts)) {
    // Violates dependences.
    LLVM_DEBUG(llvm::dbgs() << "Shifts invalid - unexpected\n";);
    return;
  }

  if (failed(affineForOpBodySkew(forOp, shifts))) {
    LLVM_DEBUG(llvm::dbgs() << "op body skewing failed - unexpected\n";);
    return;
  }
}
