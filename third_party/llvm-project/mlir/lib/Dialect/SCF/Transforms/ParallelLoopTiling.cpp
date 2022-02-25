//===- ParallelLoopTiling.cpp - Tiles scf.parallel ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop tiling on parallel loops.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace mlir::scf;

/// Tile a parallel loop of the form
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4, %arg5)
///
/// into
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4*tileSize[0],
///                                                  %arg5*tileSize[1])
///     scf.parallel (%j0, %j1) = (0, 0) to (min(%arg4*tileSize[0], %arg2-%i0)
///                                          min(%arg5*tileSize[1], %arg3-%i1))
///                                      step (%arg4, %arg5)
///
/// or, when no-min-max-bounds is true, into
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4*tileSize[0],
///                                                  %arg5*tileSize[1])
///     scf.parallel (%j0, %j1) = (0, 0) to (%arg4*tileSize[0],
///                                          %arg5*tileSize[1])
///                                      step (%arg4, %arg5)
///        %inbound = (%j0 * %arg4 + %i0 < %arg2) &&
///                   (%j1 * %arg5 + %i1 < %arg3)
///        scf.if (%inbound)
///          ....
///
/// where the uses of %i0 and %i1 in the loop body are replaced by
/// %i0 + j0 and %i1 + %j1.
//
/// The old loop is replaced with the new one.
std::pair<ParallelOp, ParallelOp>
mlir::scf::tileParallelLoop(ParallelOp op, ArrayRef<int64_t> tileSizes,
                            bool noMinMaxBounds) {
  OpBuilder b(op);
  auto zero = b.create<ConstantIndexOp>(op.getLoc(), 0);
  SmallVector<Value, 2> tileSizeConstants;
  tileSizeConstants.reserve(op.upperBound().size());
  for (size_t i = 0, end = op.upperBound().size(); i != end; ++i) {
    if (i < tileSizes.size())
      tileSizeConstants.push_back(
          b.create<ConstantIndexOp>(op.getLoc(), tileSizes[i]));
    else
      // Just pick 1 for the remaining dimensions.
      tileSizeConstants.push_back(b.create<ConstantIndexOp>(op.getLoc(), 1));
  }

  // Create the outer loop with adjusted steps.
  SmallVector<Value, 2> newSteps;
  newSteps.reserve(op.step().size());
  for (auto step : llvm::zip(op.step(), tileSizeConstants)) {
    newSteps.push_back(
        b.create<MulIOp>(op.getLoc(), std::get<0>(step), std::get<1>(step)));
  }
  auto outerLoop = b.create<ParallelOp>(op.getLoc(), op.lowerBound(),
                                        op.upperBound(), newSteps);
  b.setInsertionPointToStart(outerLoop.getBody());

  // Compute min(size, dim - offset) to avoid out-of-bounds accesses.
  auto minMap = AffineMap::get(
      /*dimCount=*/3, /*symbolCount=*/0,
      {getAffineDimExpr(/*position=*/0, b.getContext()),
       getAffineDimExpr(/*position=*/1, b.getContext()) -
           getAffineDimExpr(/*position=*/2, b.getContext())},
      b.getContext());

  // Create the inner loop with adjusted bounds.
  SmallVector<Value, 2> newBounds;
  newBounds.reserve(op.upperBound().size());
  bool needInboundCheck = false;
  for (auto dim : llvm::zip(outerLoop.lowerBound(), outerLoop.upperBound(),
                            outerLoop.step(), outerLoop.getInductionVars(),
                            op.step(), tileSizeConstants)) {
    Value lowerBound, upperBound, newStep, iv, step, tileSizeConstant;
    std::tie(lowerBound, upperBound, newStep, iv, step, tileSizeConstant) = dim;
    // Collect the statically known loop bounds
    auto lowerBoundConstant =
        dyn_cast_or_null<ConstantIndexOp>(lowerBound.getDefiningOp());
    auto upperBoundConstant =
        dyn_cast_or_null<ConstantIndexOp>(upperBound.getDefiningOp());
    auto stepConstant = dyn_cast_or_null<ConstantIndexOp>(step.getDefiningOp());
    auto tileSize =
        cast<ConstantIndexOp>(tileSizeConstant.getDefiningOp()).getValue();
    // If the loop bounds and the loop step are constant and if the number of
    // loop iterations is an integer multiple of the tile size, we use a static
    // bound for the inner loop.
    if (lowerBoundConstant && upperBoundConstant && stepConstant) {
      auto numIterations = llvm::divideCeil(upperBoundConstant.getValue() -
                                                lowerBoundConstant.getValue(),
                                            stepConstant.getValue());
      if (numIterations % tileSize == 0) {
        newBounds.push_back(newStep);
        continue;
      }
    }

    // For InboundCheck mode, just use the variable outer step
    if (noMinMaxBounds) {
      newBounds.push_back(newStep);
      needInboundCheck = true;
      continue;
    }

    // Otherwise, we dynamically compute the bound for
    // each iteration of the outer loop.
    newBounds.push_back(
        b.create<AffineMinOp>(op.getLoc(), b.getIndexType(), minMap,
                              ValueRange{newStep, upperBound, iv}));
  }
  auto innerLoop = b.create<ParallelOp>(
      op.getLoc(), SmallVector<Value, 2>(newBounds.size(), zero), newBounds,
      op.step());

  if (noMinMaxBounds && needInboundCheck) {
    b.setInsertionPointToStart(innerLoop.getBody());
    // Insert in-bound check
    Value inbound =
        b.create<ConstantOp>(op.getLoc(), b.getIntegerType(1),
                             b.getIntegerAttr(b.getIntegerType(1), 1));
    for (auto dim :
         llvm::zip(outerLoop.upperBound(), outerLoop.getInductionVars(),
                   innerLoop.getInductionVars(), innerLoop.step())) {
      Value outerUpperBound, outerIV, innerIV, innerStep;
      std::tie(outerUpperBound, outerIV, innerIV, innerStep) = dim;
      // %in_bound = %in_bound &&
      //             (%inner_iv * %inner_step + %outer_iv < %outer_upper_bound)
      Value index = b.create<AddIOp>(
          op.getLoc(), b.create<MulIOp>(op.getLoc(), innerIV, innerStep),
          outerIV);
      Value dimInbound = b.create<CmpIOp>(op.getLoc(), CmpIPredicate::ult,
                                          index, outerUpperBound);
      inbound = b.create<AndOp>(op.getLoc(), inbound, dimInbound);
    }
    auto ifInbound = b.create<IfOp>(op.getLoc(),
                                    /*resultTypes*/ ArrayRef<Type>{}, inbound,
                                    /*hasElseRegion*/ false);
    ifInbound.thenRegion().takeBody(op.region());
    Block &thenBlock = ifInbound.thenRegion().front();
    b.setInsertionPointToStart(innerLoop.getBody());
    for (auto ivs : llvm::enumerate(llvm::zip(innerLoop.getInductionVars(),
                                              outerLoop.getInductionVars()))) {
      AddIOp newIndex = b.create<AddIOp>(op.getLoc(), std::get<0>(ivs.value()),
                                         std::get<1>(ivs.value()));
      thenBlock.getArgument(ivs.index())
          .replaceAllUsesExcept(newIndex, newIndex);
    }
    thenBlock.eraseArguments(llvm::to_vector<4>(
        llvm::seq((unsigned)0, thenBlock.getNumArguments())));
  } else {
    innerLoop.region().takeBody(op.region());
    b.setInsertionPointToStart(innerLoop.getBody());
    for (auto ivs : llvm::zip(innerLoop.getInductionVars(),
                              outerLoop.getInductionVars())) {
      Value innerIndex = std::get<0>(ivs);
      AddIOp newIndex =
          b.create<AddIOp>(op.getLoc(), std::get<0>(ivs), std::get<1>(ivs));
      innerIndex.replaceAllUsesExcept(newIndex, newIndex);
    }
  }

  op.erase();
  return std::make_pair(outerLoop, innerLoop);
}

namespace {
struct ParallelLoopTiling
    : public SCFParallelLoopTilingBase<ParallelLoopTiling> {
  ParallelLoopTiling() = default;
  explicit ParallelLoopTiling(ArrayRef<int64_t> tileSizes,
                              bool noMinMaxBounds = false) {
    this->tileSizes = tileSizes;
    this->noMinMaxBounds = noMinMaxBounds;
  }

  void runOnFunction() override {
    SmallVector<ParallelOp, 2> innermostPloops;
    getInnermostParallelLoops(getFunction().getOperation(), innermostPloops);
    for (ParallelOp ploop : innermostPloops) {
      // FIXME: Add reduction support.
      if (ploop.getNumReductions() == 0)
        tileParallelLoop(ploop, tileSizes, noMinMaxBounds);
    }
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::createParallelLoopTilingPass(ArrayRef<int64_t> tileSizes,
                                   bool noMinMaxBounds) {
  return std::make_unique<ParallelLoopTiling>(tileSizes, noMinMaxBounds);
}
