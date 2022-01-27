//===- LinalgInterfaceImpl.cpp - Linalg Impl. of BufferizableOpInterface --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/LinalgInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace linalg;
using namespace comprehensive_bufferize;

namespace {

// TODO: Ops in the linalg dialect can directly implement this interface.

/// Generic conversion for any LinalgOp on tensors.
static LogicalResult bufferizeLinalgOp(RewriterBase &rewriter, LinalgOp op,
                                       const BufferizationState &state) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (op.hasBufferSemantics())
    return success();

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasTensorSemantics())
    return op->emitError() << "op does not have tensor semantics";

  // New input operands for the cloned op.
  SmallVector<Value> newInputBuffers;
  newInputBuffers.reserve(op.getNumInputs());
  for (OpOperand *opOperand : op.getInputOperands()) {
    if (op.isScalar(opOperand)) {
      newInputBuffers.push_back(opOperand->get());
      continue;
    }
    // Input operands are never written to.
    newInputBuffers.push_back(
        *state.getBuffer(rewriter, *opOperand, /*forceInPlace=*/true));
  }

  // New output operands for the cloned op.
  SmallVector<Value> newOutputBuffers;
  for (OpResult opResult : op->getOpResults()) {
    SmallVector<OpOperand *> aliasingOpOperands =
        state.getAliasingOpOperand(opResult);
    assert(aliasingOpOperands.size() == 1 && "expected 1 OpOperand");
    FailureOr<Value> resultBuffer =
        state.getBuffer(rewriter, *aliasingOpOperands.front());
    if (failed(resultBuffer))
      return failure();
    newOutputBuffers.push_back(*resultBuffer);
  }

  // Merge input/output operands.
  SmallVector<Value> newOperands = newInputBuffers;
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);
  // Clone the op, but use the new operands. Move the existing block into the
  // new op. Since the new op does not have any tensor results, it does not
  // return anything.
  assert(op->getNumRegions() == 1 && "expected that op has 1 region");
  auto newOp = cast<LinalgOp>(op.cloneWithoutRegions(
      rewriter, op.getLoc(), /*resultTypes=*/TypeRange{}, newOperands));
  rewriter.inlineRegionBefore(op->getRegion(0), newOp->getRegion(0),
                              newOp->getRegion(0).begin());

  // Replace the results of the old op with the new output buffers.
  replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);

  return success();
}

/// Linalg OpResults usually bufferize inplace with their tied (output
/// OpOperands. However, if an output OpOperand is not used in the computation,
/// it is better to bufferize inplace with an actually used input OpOperand;
/// less memory will be touched that way.
///
/// Example:
/// O(i, j) = A(i, j) + B(j)  --> bufferizes inplace to:  A(i, j) += B(j)
///
/// O(i, j) = A(j, i) + B(j)  --> cannot bufferize inplace with A because
///                               indexing maps are not identical
///
/// O(i, j) += A(i, j) + B(j) --> Output is used in computation.
/// This could bufferize inplace with A:
/// A(i, j) += O(i, j) + B(j)
/// However, we choose to bufferize inplace with O here, as there is no clear
/// benefit of choosing A. TODO: We may want to consider both options and make
/// an informed decision during analysis in the future.
static DenseMap<OpOperand *, OpResult> computeAliasingPairs(LinalgOp op) {
  DenseMap<OpOperand *, OpResult> mapping;
  for (OpResult opResult : op->getOpResults()) {
    OpOperand *tiedOperand =
        op.getOutputTensorOperands()[opResult.getResultNumber()];
    AffineMap outputIndexingMap = op.getTiedIndexingMap(tiedOperand);
    bool onlyParallelIterators = op.getNumParallelLoops() == op.getNumLoops();
    bool tiedOperandUsed = op.payloadUsesValueFromOperand(tiedOperand);

    // If the output arg is used in the computation or at least one iterator is
    // not parallel, try to bufferize inplace with the corresponding output
    // tensor.
    if (tiedOperandUsed || !onlyParallelIterators) {
      mapping[tiedOperand] = opResult;
      continue;
    }

    // Otherwise, try to bufferize inplace with one of the inputs.
    OpOperand *chosenOperand = nullptr;
    for (OpOperand *opOperand : op.getInputTensorOperands()) {
      if (opOperand->get().getType() != opResult.getType())
        continue;
      if (!op.payloadUsesValueFromOperand(opOperand))
        continue;
      if (op.getTiedIndexingMap(opOperand) != outputIndexingMap)
        continue;
      // No other OpResult bufferizes aliases with this OpOperand.
      if (mapping.count(opOperand))
        continue;
      assert(op.getTiedIndexingMap(opOperand).isProjectedPermutation() &&
             "expected projected permutation");
      chosenOperand = opOperand;
      break;
    }

    // No suitable input tensor found. Use output tensor.
    // TODO: This operand could bufferize inplace with OpOperands that have the
    // correct type, even if they are not used inside the computation.
    if (!chosenOperand)
      chosenOperand = tiedOperand;

    mapping[chosenOperand] = opResult;
  }
  return mapping;
}

/// Bufferization of linalg.generic. Replace with a new linalg.generic that
/// operates entirely on memrefs.
template <typename OpTy>
struct LinalgOpInterface
    : public BufferizableOpInterface::ExternalModel<LinalgOpInterface<OpTy>,
                                                    OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    // Operand is read if it is used in the computation.
    auto genericOp = cast<linalg::LinalgOp>(op);
    return genericOp.payloadUsesValueFromOperand(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    // Operand is written to if it has an aliasing OpResult. For more details,
    // see `computeAliasingPairs`.
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    return static_cast<bool>(
        bufferizableOp.getAliasingOpResult(opOperand, state));
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const BufferizationState &state) const {
    auto genericOp = cast<linalg::LinalgOp>(op);

    // Aliasing OpOperand/OpResult pairs are computed by `computeAliasingPairs`.
    DenseMap<OpOperand *, OpResult> pairs = computeAliasingPairs(genericOp);
    for (OpOperand *opOperand : genericOp.getInputAndOutputOperands())
      if (pairs[opOperand] == opResult)
        return {opOperand};
    return {};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    auto genericOp = cast<linalg::LinalgOp>(op);

    // Aliasing OpOperand/OpResult pairs are computed by `computeAliasingPairs`.
    DenseMap<OpOperand *, OpResult> pairs = computeAliasingPairs(genericOp);
    return pairs[&opOperand];
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationAliasInfo &aliasInfo,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    return bufferizeLinalgOp(rewriter, cast<LinalgOp>(op), state);
  }
};

struct InitTensorOpInterface
    : public BufferizableOpInterface::ExternalModel<InitTensorOpInterface,
                                                    linalg::InitTensorOp> {
  bool isMemoryWrite(Operation *op, OpResult opResult,
                     const BufferizationState &state) const {
    // InitTensorOps allocate but do not write.
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto initTensorOp = cast<linalg::InitTensorOp>(op);

    // The InitTensorOp may have been eliminated.
    if (initTensorOp->getUses().empty())
      return success();

    FailureOr<Value> alloc = state.createAlloc(
        rewriter, initTensorOp->getLoc(), initTensorOp.result(),
        state.getOptions().createDeallocs);
    if (failed(alloc))
      return failure();
    replaceOpWithBufferizedValues(rewriter, op, *alloc);
    return success();
  }
};

/// Bufferization of linalg.tiled_loop. Replace with a new linalg.tiled_loop
/// that operates entirely on memrefs.
struct TiledLoopOpInterface
    : public BufferizableOpInterface::ExternalModel<TiledLoopOpInterface,
                                                    linalg::TiledLoopOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    auto tiledLoopOp = cast<linalg::TiledLoopOp>(op);

    // linalg.tiled_loop operands alone do not bufferize to a memory read, but
    // one of the uses of their matching bbArgs may.
    return state.isValueRead(tiledLoopOp.getTiedBlockArgument(opOperand));
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    auto bufferizableOp = cast<BufferizableOpInterface>(op);

    // Only operands with an aliasing OpResult (i.e., output operands) bufferize
    // to a memory write.
    return static_cast<bool>(
        bufferizableOp.getAliasingOpResult(opOperand, state));
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    auto tiledLoopOp = cast<linalg::TiledLoopOp>(op);

    // Output operands are tied to their corresponding OpResults.
    return tiledLoopOp.getTiedOpResult(opOperand);
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationAliasInfo &aliasInfo,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  bool isWritable(Operation *op, Value value,
                  const BufferizationState &state) const {
    // Interestingly, linalg::TiledLoopOp's bbArgs can **always** be viewed
    // inplace from the perspective of nested ops:
    //   1. Either the matching iter operand is not bufferized inplace and an
    //      alloc + optional copy makes the bbArg itself inplaceable.
    //   2. Or the matching iter operand is bufferized inplace and bbArg just
    //      bufferizes to that too.
    return true;
  }

  bool isAllocationHoistingBarrier(Operation *op) const { return true; }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto tiledLoopOp = cast<linalg::TiledLoopOp>(op);

    // Compute new inputs, outputs and results.
    SmallVector<Value> newInputs, newOutputs, newResults;
    for (unsigned i = tiledLoopOp.getNumControlOperands();
         i < tiledLoopOp->getNumOperands(); ++i) {
      OpOperand &operand = tiledLoopOp->getOpOperand(i);
      Value rewrittenValue = operand.get();
      if (rewrittenValue.getType().isa<TensorType>()) {
        FailureOr<Value> bufferOrFailure = state.getBuffer(rewriter, operand);
        if (failed(bufferOrFailure))
          return failure();
        rewrittenValue = *bufferOrFailure;
      }
      if (i <
          tiledLoopOp.getNumControlOperands() + tiledLoopOp.getNumInputs()) {
        newInputs.push_back(rewrittenValue);
      } else {
        newOutputs.push_back(rewrittenValue);
        if (operand.get().getType().isa<TensorType>())
          newResults.push_back(rewrittenValue);
      }
    }

    // Create new TiledLoopOp.
    auto newTiledLoopOp = rewriter.create<TiledLoopOp>(
        tiledLoopOp.getLoc(), tiledLoopOp.lowerBound(),
        tiledLoopOp.upperBound(), tiledLoopOp.step(), newInputs, newOutputs,
        tiledLoopOp.iterator_types(), tiledLoopOp.distribution_types());

    // Remove terminator.
    if (!newTiledLoopOp.getBody()->empty())
      rewriter.eraseOp(tiledLoopOp.getBody()->getTerminator());

    // Compute new loop body arguments.
    SmallVector<Value> newBlockArgs, newRegionInOutArgs, oldRegionInOutArgs;
    ValueRange newInductionVars = newTiledLoopOp.getInductionVars();
    newBlockArgs.append(newInductionVars.begin(), newInductionVars.end());

    ValueRange newRegionInArgs = newTiledLoopOp.getRegionInputArgs();
    ValueRange newRegionOutArgs = newTiledLoopOp.getRegionOutputArgs();
    newRegionInOutArgs.append(newRegionInArgs.begin(), newRegionInArgs.end());
    newRegionInOutArgs.append(newRegionOutArgs.begin(), newRegionOutArgs.end());

    ValueRange oldRegionInArgs = tiledLoopOp.getRegionInputArgs();
    ValueRange oldRegionOutArgs = tiledLoopOp.getRegionOutputArgs();
    oldRegionInOutArgs.append(oldRegionInArgs.begin(), oldRegionInArgs.end());
    oldRegionInOutArgs.append(oldRegionOutArgs.begin(), oldRegionOutArgs.end());
    assert(newRegionInArgs.size() == oldRegionInArgs.size() &&
           "expected same number of input args");
    assert(newRegionOutArgs.size() == oldRegionOutArgs.size() &&
           "expected same number of output args");

    for (auto it : llvm::zip(oldRegionInOutArgs, newRegionInOutArgs)) {
      Value oldArg = std::get<0>(it);
      Value newArg = std::get<1>(it);
      rewriter.setInsertionPointToStart(newTiledLoopOp.getBody());
      if (oldArg.getType().isa<TensorType>()) {
        newBlockArgs.push_back(rewriter.create<bufferization::ToTensorOp>(
            oldArg.getLoc(), newArg));
      } else {
        newBlockArgs.push_back(newArg);
      }
    }

    // Move old body into new loop.
    rewriter.mergeBlocks(tiledLoopOp.getBody(), newTiledLoopOp.getBody(),
                         newBlockArgs);

    // Replace previous terminator with a new one that does not yield anything.
    auto oldTerminator =
        cast<linalg::YieldOp>(newTiledLoopOp.getBody()->getTerminator());
    rewriter.setInsertionPointToEnd(newTiledLoopOp.getBody());
    auto newTerminator =
        rewriter.create<linalg::YieldOp>(oldTerminator->getLoc());

    // Copy buffer of yielded tensor to output buffer. If everything bufferized
    // inplace, this copy will fold away.
    rewriter.setInsertionPoint(newTerminator);
    for (auto it : llvm::zip(oldTerminator.values(), newOutputs)) {
      Value output = std::get<1>(it);
      Value toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(
          newTerminator.getLoc(), output.getType(), std::get<0>(it));
      state.createMemCpy(rewriter, newTerminator.getLoc(), toMemrefOp, output);
    }

    // Erase old terminator.
    rewriter.eraseOp(oldTerminator);

    // Replace results and delete old op.
    replaceOpWithBufferizedValues(rewriter, op, newResults);

    return success();
  }
};

/// Bufferization of linalg.yield. Bufferized as part of linalg.tiled_loop's
/// bufferization.
struct YieldOpInterface
    : public BufferizableOpInterface::ExternalModel<YieldOpInterface,
                                                    linalg::YieldOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return OpResult();
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const BufferizationState &state) const {
    // Yield operands always bufferize inplace. Otherwise, an alloc + copy
    // may be generated inside the block. We should not return/yield allocations
    // when possible.
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto yieldOp = cast<linalg::YieldOp>(op);

    if (!yieldOp->getParentOfType<TiledLoopOp>())
      return yieldOp->emitError(
          "expected that linalg.yield terminates a tiled_loop");

    assert(yieldOp->getOpOperands().empty() &&
           "expected that linalg.yield was bufferized together with"
           " tiled_loop");
    return success();
  }
};

/// Helper structure that iterates over all LinalgOps in `OpTys` and registers
/// the `BufferizableOpInterface` with each of them.
template <typename... OpTys>
struct LinalgOpInterfaceHelper;

template <typename First, typename... Others>
struct LinalgOpInterfaceHelper<First, Others...> {
  static void registerOpInterface(DialectRegistry &registry) {
    registry.addOpInterface<First, LinalgOpInterface<First>>();
    LinalgOpInterfaceHelper<Others...>::registerOpInterface(registry);
  }
};

template <>
struct LinalgOpInterfaceHelper<> {
  static void registerOpInterface(DialectRegistry &registry) {}
};

} // namespace

/// Try to eliminate InitTensorOps inside `op`. An InitTensorOp is replaced
/// with the the result of `rewriteFunc` if it is anchored on a matching
/// OpOperand. "Anchored" means that there is a path on the reverse SSA use-def
/// chain, starting from the OpOperand and always following the aliasing
/// OpOperand, that eventually ends at a single InitTensorOp.
LogicalResult
mlir::linalg::comprehensive_bufferize::linalg_ext::InitTensorEliminationStep::
    eliminateInitTensors(Operation *op, BufferizationState &state,
                         BufferizationAliasInfo &aliasInfo,
                         AnchorMatchFn anchorMatchFunc, RewriteFn rewriteFunc,
                         SmallVector<Operation *> &newOps) {
  OpBuilder b(op->getContext());

  WalkResult status = op->walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      // Skip operands that do not bufferize inplace.
      if (!aliasInfo.isInPlace(operand))
        continue;
      // Is this a matching OpOperand?
      if (!anchorMatchFunc(operand))
        continue;
      SetVector<Value> maybeInitTensor =
          state.findValueInReverseUseDefChain(operand.get(), [&](Value val) {
            // Continue traversal until this function returns true.
            OpResult opResult = val.dyn_cast<OpResult>();
            if (!opResult)
              return true;
            SmallVector<OpOperand *> opOperands =
                state.getAliasingOpOperand(opResult);
            if (!llvm::all_of(opOperands, [&](OpOperand *operand) {
                  return aliasInfo.isInPlace(*operand);
                }))
              return true;
            // Only equivalent tensors are supported at the moment.
            // TODO: Support cases such as extract_slice(init_tensor)
            return !llvm::all_of(opOperands, [&](OpOperand *operand) {
              return aliasInfo.areEquivalentBufferizedValues(operand->get(),
                                                             opResult);
            });
          });

      // Replace only if the reverse use-def chain ends at exactly one
      // InitTensorOp.
      if (maybeInitTensor.size() != 1 ||
          !maybeInitTensor.front().getDefiningOp<InitTensorOp>())
        return WalkResult::skip();
      Value initTensor = maybeInitTensor.front();

      // Create a replacement for the InitTensorOp.
      b.setInsertionPoint(initTensor.getDefiningOp());
      Value replacement = rewriteFunc(b, initTensor.getLoc(), operand);
      if (!replacement)
        continue;

      // Uses of the InitTensorOp are replaced here, but the op is not deleted.
      // InitTensorOps without uses are ignored by the bufferization.
      initTensor.replaceAllUsesWith(replacement);
      aliasInfo.createAliasInfoEntry(replacement);
      aliasInfo.unionAliasSets(initTensor, replacement);
      aliasInfo.unionEquivalenceClasses(initTensor, replacement);

      // Register replacement ops.
      if (Operation *newOp = replacement.getDefiningOp())
        newOps.push_back(newOp);
    }

    // Advance to the next operation.
    return WalkResult::advance();
  });

  return failure(status.wasInterrupted());
}

/// Try to eliminate InitTensorOps inside `op`. An InitTensorOp can be
/// eliminated if it is eventually inserted into another tensor (and some other
/// conditions are met).
///
/// E.g.:
/// %0 = linalg.init_tensor
/// %1 = linalg.fill(%cst, %0) {inplace = [true]}
/// %2 = tensor.insert_slice %1 into %t[10][20][1]
///
/// InitTensorOp elimination will try to fill %t inplace instead of filling a
/// new allocation %0 and inserting it into %t. This is done by replacing the
/// InitTensorOp with:
///
/// %0 = tensor.extract_slice %t[10][20][1]
///
/// The analysis looks for matching ExtractSliceOp/InsertSliceOp pairs and lets
/// those bufferize inplace in the absence of other conflicts.
///
/// Starting from an InsertSliceOp, an InitTensorOp at the end of the insert
/// source's reverse use-def chain is eliminated if:
/// * The InsertSliceOp was decided to bufferize inplace.
/// * On the reverse use-def chain path from the InsertSliceOp to the
///   InitTensorOp, all ops were decided to bufferize inplace and the buffer
///   relation is "equivalent" (TODO: can be relaxed if needed).
/// * The reverse use-def chain has exactly one end, which is the InitTensorOp.
///
/// Note that the newly inserted ExtractSliceOp may have to bufferize
/// out-of-place due to RaW conflicts.
LogicalResult mlir::linalg::comprehensive_bufferize::linalg_ext::
    InsertSliceAnchoredInitTensorEliminationStep::run(
        Operation *op, BufferizationState &state,
        BufferizationAliasInfo &aliasInfo, SmallVector<Operation *> &newOps) {
  return eliminateInitTensors(
      op, state, aliasInfo,
      /*anchorMatchFunc=*/
      [&](OpOperand &operand) {
        auto insertSliceOp =
            dyn_cast<tensor::InsertSliceOp>(operand.getOwner());
        if (!insertSliceOp)
          return false;
        // Only inplace bufferized InsertSliceOps are eligible.
        if (!aliasInfo.isInPlace(insertSliceOp->getOpOperand(1) /*dest*/))
          return false;
        return &operand == &insertSliceOp->getOpOperand(0) /*source*/;
      },
      /*rewriteFunc=*/
      [](OpBuilder &b, Location loc, OpOperand &operand) {
        auto insertOp = cast<tensor::InsertSliceOp>(operand.getOwner());
        // Expand offsets, sizes and strides to the full rank to handle the
        // rank-reducing case.
        SmallVector<OpFoldResult> mixedOffsets = insertOp.getMixedOffsets();
        SmallVector<OpFoldResult> mixedSizes = insertOp.getMixedSizes();
        SmallVector<OpFoldResult> mixedStrides = insertOp.getMixedStrides();
        OffsetSizeAndStrideOpInterface::expandToRank(
            insertOp.dest(), mixedOffsets, mixedSizes, mixedStrides,
            [&](Value target, int64_t dim) -> OpFoldResult {
              auto shapedType = target.getType().cast<ShapedType>();
              if (shapedType.isDynamicDim(dim))
                return b.create<tensor::DimOp>(loc, target, dim).result();
              return b.getIndexAttr(shapedType.getDimSize(dim));
            });
        auto t = tensor::ExtractSliceOp::inferRankReducedResultType(
            insertOp.getSourceType().getRank(),
            insertOp.dest().getType().cast<RankedTensorType>(), mixedOffsets,
            mixedSizes, mixedStrides);
        auto extractOp = b.create<tensor::ExtractSliceOp>(
            loc, t, insertOp.dest(), mixedOffsets, mixedSizes, mixedStrides);
        return extractOp.result();
      },
      newOps);
}

void mlir::linalg::comprehensive_bufferize::linalg_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<linalg::InitTensorOp, InitTensorOpInterface>();
  registry.addOpInterface<linalg::TiledLoopOp, TiledLoopOpInterface>();
  registry.addOpInterface<linalg::YieldOp, YieldOpInterface>();

  // Register all Linalg structured ops. `LinalgOp` is an interface and it is
  // not possible to attach an external interface to an existing interface.
  // Therefore, attach the `BufferizableOpInterface` to all ops one-by-one.
  LinalgOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >::registerOpInterface(registry);
}
