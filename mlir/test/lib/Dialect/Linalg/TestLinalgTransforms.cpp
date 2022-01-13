//===- TestLinalgTransforms.cpp - Test Linalg transformation patterns -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Linalg transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct TestLinalgTransforms
    : public PassWrapper<TestLinalgTransforms, FunctionPass> {
  TestLinalgTransforms() = default;
  TestLinalgTransforms(const TestLinalgTransforms &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect,
                    StandardOpsDialect,
                    vector::VectorDialect,
                    gpu::GPUDialect>();
    // clang-format on
  }
  StringRef getArgument() const final {
    return "test-linalg-transform-patterns";
  }
  StringRef getDescription() const final {
    return "Test Linalg transformation patterns by applying them greedily.";
  }

  void runOnFunction() override;

  Option<bool> testPatterns{*this, "test-patterns",
                            llvm::cl::desc("Test a mixed set of patterns"),
                            llvm::cl::init(false)};
  Option<bool> testMatmulToVectorPatterns1dTiling{
      *this, "test-matmul-to-vector-patterns-tile-1d",
      llvm::cl::desc(
          "Test a fused pass that applies patterns from matmul to vectors via "
          "1-d tiling"),
      llvm::cl::init(false)};
  Option<bool> testMatmulToVectorPatterns2dTiling{
      *this, "test-matmul-to-vector-patterns-tile-2d",
      llvm::cl::desc(
          "Test a fused pass that applies patterns from matmul to vectors via "
          "2-d tiling"),
      llvm::cl::init(false)};
  Option<bool> testPromotionOptions{*this, "test-linalg-promotion-options",
                                    llvm::cl::desc("Test promotion options"),
                                    llvm::cl::init(false)};
  Option<bool> testTileAndDistributionOptions{
      *this, "test-tile-and-distribute-options",
      llvm::cl::desc("Test tile and distribute options"),
      llvm::cl::init(false)};
  Option<bool> testVectorTransferForwardingPatterns{
      *this, "test-vector-transfer-forwarding-patterns",
      llvm::cl::desc(
          "Test a fused pass that forwards linalg.copy to vector.transfer"),
      llvm::cl::init(false)};
  Option<bool> testGenericToVectorPattern{
      *this, "test-linalg-to-vector-patterns",
      llvm::cl::desc("Test a set of patterns that rewrite a linalg contraction "
                     "in vector.contract form"),
      llvm::cl::init(false)};
  Option<bool> testTileAndPadPattern{
      *this, "test-tile-and-pad-pattern",
      llvm::cl::desc("Test tile and pad pattern"), llvm::cl::init(false)};
  Option<int> testHoistPadding{*this, "test-hoist-padding",
                               llvm::cl::desc("Test hoist padding"),
                               llvm::cl::init(0)};
  Option<bool> testTransformPadTensor{
      *this, "test-transform-pad-tensor",
      llvm::cl::desc("Test transform pad tensor by copying with generic ops"),
      llvm::cl::init(false)};
  Option<bool> testGeneralizePadTensor{
      *this, "test-generalize-pad-tensor",
      llvm::cl::desc("Test transform pad tensor by copying with generic ops"),
      llvm::cl::init(false)};
  Option<bool> testSwapSubTensorPadTensor{
      *this, "test-swap-subtensor-padtensor",
      llvm::cl::desc("Test rewrite of subtensor(pad_tensor) into "
                     "pad_tensor(subtensor)"),
      llvm::cl::init(false)};
  ListOption<int64_t> tileSizesForPadding{
      *this, "tile-sizes-for-padding",
      llvm::cl::desc("Linalg tile sizes when tile+pad"), llvm::cl::ZeroOrMore,
      llvm::cl::MiscFlags::CommaSeparated};
  ListOption<unsigned> testInterchangePattern{
      *this, "test-interchange-pattern", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Test the interchange pattern.")};
};
} // end anonymous namespace

static void applyPatterns(FuncOp funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);

  //===--------------------------------------------------------------------===//
  // Linalg tiling patterns.
  //===--------------------------------------------------------------------===//
  patterns.add<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({2000, 3000, 4000}),
      LinalgTransformationFilter(Identifier::get("MEM", ctx),
                                 Identifier::get("L3", ctx)));
  patterns.add<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({200, 300, 400}),
      LinalgTransformationFilter(Identifier::get("L3", ctx),
                                 Identifier::get("L2", ctx)));
  patterns.add<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({20, 30, 40}),
      LinalgTransformationFilter(Identifier::get("L2", ctx),
                                 Identifier::get("L1", ctx)));
  patterns.add<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({2, 3, 4}),
      LinalgTransformationFilter(Identifier::get("L1", ctx),
                                 Identifier::get("REG", ctx)));

  patterns.add<LinalgTilingPattern<MatvecOp>>(
      ctx,
      LinalgTilingOptions().setTileSizes({5, 6}).setLoopType(
          LinalgTilingLoopType::ParallelLoops),
      LinalgTransformationFilter(ArrayRef<Identifier>{},
                                 Identifier::get("L1", ctx)));

  patterns.add<LinalgTilingPattern<DotOp>>(
      ctx, LinalgTilingOptions().setTileSizes(8000),
      LinalgTransformationFilter(
          ArrayRef<Identifier>{Identifier::get("MEM", ctx),
                               Identifier::get("L3", ctx),
                               Identifier::get("L2", ctx)},
          Identifier::get("REG", ctx)));

  //===--------------------------------------------------------------------===//
  // Linalg tiling and permutation patterns.
  //===--------------------------------------------------------------------===//
  patterns.add<LinalgTilingPattern<MatmulOp>>(
      ctx,
      LinalgTilingOptions()
          .setTileSizes({2000, 3000, 4000})
          .setInterchange({1, 2, 0}),
      LinalgTransformationFilter(Identifier::get("__with_perm__", ctx),
                                 Identifier::get("L2__with_perm__", ctx)));
  patterns.add<LinalgTilingPattern<MatmulOp>>(
      ctx,
      LinalgTilingOptions()
          .setTileSizes({200, 300, 400})
          .setInterchange({1, 0, 2}),
      LinalgTransformationFilter(Identifier::get("L2__with_perm__", ctx),
                                 Identifier::get("L1__with_perm__", ctx)));
  patterns.add<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({20, 30, 40}),
      LinalgTransformationFilter(Identifier::get("L1__with_perm__", ctx),
                                 Identifier::get("REG__with_perm__", ctx)));

  patterns.add<LinalgTilingPattern<MatvecOp>>(
      ctx, LinalgTilingOptions().setTileSizes({5, 6}).setInterchange({1, 0}),
      LinalgTransformationFilter(Identifier::get("__with_perm__", ctx),
                                 Identifier::get("L1__with_perm__", ctx)));

  patterns.add<LinalgTilingPattern<MatmulOp>>(
      ctx,
      LinalgTilingOptions()
          .setTileSizes({16, 8, 4})
          .setInterchange({1, 2, 0})
          .setLoopType(LinalgTilingLoopType::ParallelLoops),
      LinalgTransformationFilter(
          Identifier::get("par__with_perm__", ctx),
          Identifier::get("after_par__with_perm__", ctx)));

  //===--------------------------------------------------------------------===//
  // Linalg to loops patterns.
  //===--------------------------------------------------------------------===//
  patterns.add<LinalgLoweringPattern<DotOp>>(
      ctx,
      /*loweringType=*/LinalgLoweringType::Loops,
      LinalgTransformationFilter(Identifier::get("REG", ctx)));

  //===--------------------------------------------------------------------===//
  // Linalg distribution patterns.
  //===--------------------------------------------------------------------===//
  LinalgLoopDistributionOptions distributionOptions;

  //===--------------------------------------------------------------------===//
  // Linalg to vector contraction patterns.
  //===--------------------------------------------------------------------===//
  patterns.add<LinalgVectorizationPattern>(
      ctx, LinalgTransformationFilter(Identifier::get("VECTORIZE", ctx))
               .addOpFilter<MatmulOp, FillOp, CopyOp, GenericOp>());

  //===--------------------------------------------------------------------===//
  // Linalg generic interchange pattern.
  //===--------------------------------------------------------------------===//
  patterns.add<GenericOpInterchangePattern>(
      ctx,
      /*interchangeVector=*/ArrayRef<unsigned>{1, 2, 0},
      LinalgTransformationFilter(ArrayRef<Identifier>{},
                                 Identifier::get("PERMUTED", ctx)));

  //===--------------------------------------------------------------------===//
  // Linalg subview operands promotion.
  //===--------------------------------------------------------------------===//
  patterns.add<LinalgPromotionPattern<MatmulOp>>(
      ctx, LinalgPromotionOptions().setUseFullTileBuffersByDefault(true),
      LinalgTransformationFilter(Identifier::get("_promote_views_", ctx),
                                 Identifier::get("_views_promoted_", ctx)));
  patterns.add<LinalgPromotionPattern<MatmulOp>>(
      ctx,
      LinalgPromotionOptions()
          .setOperandsToPromote({0})
          .setUseFullTileBuffersByDefault(true),
      LinalgTransformationFilter(
          Identifier::get("_promote_first_view_", ctx),
          Identifier::get("_first_view_promoted_", ctx)));
  patterns.add<LinalgPromotionPattern<FillOp>>(
      ctx,
      LinalgPromotionOptions()
          .setOperandsToPromote({1})
          .setUseFullTileBuffers({false, true})
          .setAlignment(32),
      LinalgTransformationFilter(
          Identifier::get("_promote_views_aligned_", ctx),
          Identifier::get("_views_aligned_promoted_", ctx)));

  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

  // Drop the marker.
  funcOp.walk([](LinalgOp op) {
    op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}

static void fillL1TilingAndMatmulToVectorPatterns(
    FuncOp funcOp, StringRef startMarker,
    SmallVectorImpl<RewritePatternSet> &patternsVector) {
  MLIRContext *ctx = funcOp.getContext();
  patternsVector.emplace_back(
      ctx, std::make_unique<LinalgTilingPattern<MatmulOp>>(
               ctx,
               LinalgTilingOptions()
                   .setTileSizes({8, 12, 16})
                   .setInterchange({1, 0, 2}),
               LinalgTransformationFilter(Identifier::get(startMarker, ctx),
                                          Identifier::get("L1", ctx))));

  patternsVector.emplace_back(
      ctx,
      std::make_unique<LinalgPromotionPattern<MatmulOp>>(
          ctx, LinalgPromotionOptions().setUseFullTileBuffersByDefault(true),
          LinalgTransformationFilter(Identifier::get("L1", ctx),
                                     Identifier::get("VEC", ctx))));

  patternsVector.emplace_back(
      ctx, std::make_unique<LinalgVectorizationPattern>(
               MatmulOp::getOperationName(), ctx, LinalgVectorizationOptions(),
               LinalgTransformationFilter(Identifier::get("VEC", ctx))));
  patternsVector.back().add<LinalgVectorizationPattern>(
      ctx, LinalgTransformationFilter().addFilter(
               [](Operation *op) { return success(isa<FillOp, CopyOp>(op)); }));
}

//===----------------------------------------------------------------------===//
// Test promotion callbacks
//===----------------------------------------------------------------------===//

// Allocation call back
static Optional<Value> allocCallBackFn(OpBuilder &b, memref::SubViewOp subView,
                                       ArrayRef<Value> boundingSubViewSize,
                                       DataLayout &layout) {
  SmallVector<int64_t, 4> shape(boundingSubViewSize.size(), -1);
  return b
      .create<memref::AllocOp>(
          subView.getLoc(),
          MemRefType::get(shape, subView.getType().getElementType(),
                          /*affineMapComposition =*/{}, 3),
          boundingSubViewSize)
      .getResult();
}

// Deallocation callback
static LogicalResult deallocCallBackFn(OpBuilder &b, Value buffer) {
  b.create<memref::DeallocOp>(buffer.getLoc(), buffer);
  return success();
}

// Copy in call back
static LogicalResult copyCallBackFn(OpBuilder &b, Value src, Value dst,
                                    bool isOutput) {
  auto floatType = src.getType().cast<MemRefType>().getElementType();
  if (!floatType.isa<FloatType>())
    return failure();
  if (!isOutput) {
    Value cst =
        b.create<ConstantOp>(src.getLoc(), FloatAttr::get(floatType, 42.0));
    b.create<FillOp>(src.getLoc(), cst, dst);
  }
  b.create<CopyOp>(src.getLoc(), src, dst);
  return success();
}

static void fillPromotionCallBackPatterns(MLIRContext *ctx,
                                          RewritePatternSet &patterns) {
  patterns.add<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({16, 16, 16}),
      LinalgTransformationFilter(Identifier::get("START", ctx),
                                 Identifier::get("PROMOTE", ctx)));
  patterns.add<LinalgPromotionPattern<MatmulOp>>(
      ctx,
      LinalgPromotionOptions()
          .setOperandsToPromote({0, 2})
          .setUseFullTileBuffers({false, false})
          .setAllocationDeallocationFns(allocCallBackFn, deallocCallBackFn)
          .setCopyInOutFns(
              [](OpBuilder &b, Value src, Value dst) -> LogicalResult {
                return copyCallBackFn(b, src, dst, false);
              },
              [](OpBuilder &b, Value src, Value dst) -> LogicalResult {
                return copyCallBackFn(b, src, dst, true);
              }),
      LinalgTransformationFilter(Identifier::get("PROMOTE", ctx)));
}

template <typename IdOp, typename NProcsOp>
static SmallVector<ProcInfo, 2>
getGpuProcIds(OpBuilder &b, Location loc, ArrayRef<Range> parallelLoopRanges) {
  size_t count = std::min<size_t>(3, parallelLoopRanges.size());
  SmallVector<ProcInfo, 2> procInfo(count);
  const char *xyz[] = {"x", "y", "z"};
  Type indexType = b.getIndexType();
  for (unsigned i = 0; i < count; ++i) {
    procInfo[count - 1 - i] = {
        b.create<IdOp>(loc, indexType, b.getStringAttr(xyz[i])),
        b.create<NProcsOp>(loc, indexType, b.getStringAttr(xyz[i]))};
  }
  return procInfo;
}

static void fillTileAndDistributePatterns(MLIRContext *context,
                                          RewritePatternSet &patterns) {
  {
    LinalgLoopDistributionOptions cyclicNprocsEqNiters;
    cyclicNprocsEqNiters.distributionMethod.resize(
        2, DistributionMethod::CyclicNumProcsEqNumIters);
    cyclicNprocsEqNiters.procInfo =
        getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;
    patterns.add<LinalgTilingPattern<MatmulOp>>(
        context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::ParallelLoops)
            .setDistributionOptions(cyclicNprocsEqNiters),
        LinalgTransformationFilter(
            Identifier::get("distribute1", context),
            Identifier::get("after_distribute1", context)));
  }

  {
    LinalgLoopDistributionOptions cyclicNprocsGeNiters;
    cyclicNprocsGeNiters.distributionMethod.resize(
        2, DistributionMethod::CyclicNumProcsGeNumIters);
    cyclicNprocsGeNiters.procInfo =
        getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;
    patterns.add<LinalgTilingPattern<MatmulOp>>(
        context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::ParallelLoops)
            .setDistributionOptions(cyclicNprocsGeNiters),
        LinalgTransformationFilter(
            Identifier::get("distribute2", context),
            Identifier::get("after_distribute2", context)));
  }

  {
    LinalgLoopDistributionOptions cyclicNprocsDefault;
    cyclicNprocsDefault.distributionMethod.resize(2,
                                                  DistributionMethod::Cyclic);
    cyclicNprocsDefault.procInfo =
        getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;
    patterns.add<LinalgTilingPattern<MatmulOp>>(
        context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::ParallelLoops)
            .setDistributionOptions(cyclicNprocsDefault),
        LinalgTransformationFilter(
            Identifier::get("distribute3", context),
            Identifier::get("after_distribute3", context)));
  }

  {
    LinalgLoopDistributionOptions cyclicNprocsMixed1;
    cyclicNprocsMixed1.distributionMethod = {
        DistributionMethod::CyclicNumProcsEqNumIters,
        DistributionMethod::CyclicNumProcsGeNumIters};
    cyclicNprocsMixed1.procInfo = getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;
    patterns.add<LinalgTilingPattern<MatmulOp>>(
        context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::ParallelLoops)
            .setDistributionOptions(cyclicNprocsMixed1),
        LinalgTransformationFilter(
            Identifier::get("distribute4", context),
            Identifier::get("after_distribute4", context)));
  }

  {
    LinalgLoopDistributionOptions cyclicNprocsMixed2;
    cyclicNprocsMixed2.distributionMethod = {
        DistributionMethod::CyclicNumProcsGeNumIters,
        DistributionMethod::Cyclic};
    cyclicNprocsMixed2.procInfo = getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;
    patterns.add<LinalgTilingPattern<MatmulOp>>(
        context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::ParallelLoops)
            .setDistributionOptions(cyclicNprocsMixed2),
        LinalgTransformationFilter(
            Identifier::get("distribute5", context),
            Identifier::get("after_distribute5", context)));
  }

  {
    LinalgLoopDistributionOptions cyclicNprocsMixed3;
    cyclicNprocsMixed3.distributionMethod = {
        DistributionMethod::Cyclic,
        DistributionMethod::CyclicNumProcsEqNumIters};
    cyclicNprocsMixed3.procInfo = getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;

    patterns.add<LinalgTilingPattern<MatmulOp>>(
        context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::ParallelLoops)
            .setDistributionOptions(cyclicNprocsMixed3),
        LinalgTransformationFilter(
            Identifier::get("distribute6", context),
            Identifier::get("after_distribute6", context)));
  }

  {
    LinalgLoopDistributionOptions cyclicNprocsEqNiters;
    cyclicNprocsEqNiters.distributionMethod.resize(2,
                                                   DistributionMethod::Cyclic);
    cyclicNprocsEqNiters.procInfo =
        getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;
    patterns.add<LinalgTilingPattern<MatmulOp>>(
        context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::Loops)
            .setDistributionOptions(cyclicNprocsEqNiters),
        LinalgTransformationFilter(
            Identifier::get("tensors_distribute1", context),
            Identifier::get("tensors_after_distribute1", context)));
  }
}

static void
applyMatmulToVectorPatterns(FuncOp funcOp,
                            bool testMatmulToVectorPatterns1dTiling,
                            bool testMatmulToVectorPatterns2dTiling) {
  MLIRContext *ctx = funcOp.getContext();
  SmallVector<RewritePatternSet, 4> stage1Patterns;
  if (testMatmulToVectorPatterns1dTiling) {
    fillL1TilingAndMatmulToVectorPatterns(funcOp, Identifier::get("START", ctx),
                                          stage1Patterns);
  } else if (testMatmulToVectorPatterns2dTiling) {
    stage1Patterns.emplace_back(
        ctx, std::make_unique<LinalgTilingPattern<MatmulOp>>(
                 ctx,
                 LinalgTilingOptions()
                     .setTileSizes({768, 264, 768})
                     .setInterchange({1, 2, 0}),
                 LinalgTransformationFilter(Identifier::get("START", ctx),
                                            Identifier::get("L2", ctx))));
    fillL1TilingAndMatmulToVectorPatterns(funcOp, Identifier::get("L2", ctx),
                                          stage1Patterns);
  }
  SmallVector<FrozenRewritePatternSet, 4> frozenStage1Patterns;
  llvm::move(stage1Patterns, std::back_inserter(frozenStage1Patterns));
  FrozenRewritePatternSet stage2Patterns =
      getLinalgTilingCanonicalizationPatterns(ctx);
  (void)applyStagedPatterns(funcOp, frozenStage1Patterns,
                            std::move(stage2Patterns));
}

static void applyVectorTransferForwardingPatterns(FuncOp funcOp) {
  RewritePatternSet forwardPattern(funcOp.getContext());
  forwardPattern.add<LinalgCopyVTRForwardingPattern>(funcOp.getContext());
  forwardPattern.add<LinalgCopyVTWForwardingPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(forwardPattern));
}

static void applyLinalgToVectorPatterns(FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<LinalgVectorizationPattern>(
      funcOp.getContext(),
      LinalgTransformationFilter()
          .addOpFilter<ContractionOpInterface, FillOp, CopyOp, GenericOp>());
  populatePadTensorOpVectorizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyPadTensorToGenericPatterns(FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<PadTensorOpTransformationPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyGeneralizePadTensorPatterns(FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<GeneralizePadTensorOpPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyExtractSliceOfPadTensorSwapPattern(FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<ExtractSliceOfPadTensorSwapPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

// For now, just assume it is the zero of type.
// In the future, it should be the zero of type + op.
static Value getNeutralOfLinalgOp(OpBuilder &b, OpOperand &op) {
  auto t = getElementTypeOrSelf(op.get());
  return b.create<ConstantOp>(op.getOwner()->getLoc(), t, b.getZeroAttr(t));
}

static void applyTileAndPadPattern(FuncOp funcOp, ArrayRef<int64_t> tileSizes) {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet tilingPattern(context);
  auto linalgTilingOptions =
      linalg::LinalgTilingOptions()
          .setTileSizes(tileSizes)
          .setPaddingValueComputationFunction(getNeutralOfLinalgOp);
  tilingPattern.add<linalg::LinalgTilingPattern<linalg::MatmulI8I8I32Op>,
                    linalg::LinalgTilingPattern<linalg::GenericOp>>(
      context, linalgTilingOptions,
      linalg::LinalgTransformationFilter(
          Identifier::get("tile-and-pad", context)));
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(tilingPattern));
}

static void applyInterchangePattern(FuncOp funcOp,
                                    ArrayRef<unsigned> interchangeVector) {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet interchangePattern(context);
  interchangePattern.add<GenericOpInterchangePattern>(
      context, interchangeVector,
      LinalgTransformationFilter(ArrayRef<Identifier>{},
                                 Identifier::get("interchange", context)));
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(interchangePattern));
}

/// Apply transformations specified as patterns.
void TestLinalgTransforms::runOnFunction() {
  auto lambda = [&](void *) {
    getFunction().walk([](LinalgOp op) {
      op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
    });
  };
  std::unique_ptr<void, decltype(lambda)> cleanupGuard{(void *)1, lambda};

  if (testPromotionOptions) {
    RewritePatternSet patterns(&getContext());
    fillPromotionCallBackPatterns(&getContext(), patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
    return;
  }
  if (testTileAndDistributionOptions) {
    RewritePatternSet patterns(&getContext());
    fillTileAndDistributePatterns(&getContext(), patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
    return;
  }
  if (testPatterns)
    return applyPatterns(getFunction());
  if (testMatmulToVectorPatterns1dTiling || testMatmulToVectorPatterns2dTiling)
    return applyMatmulToVectorPatterns(getFunction(),
                                       testMatmulToVectorPatterns1dTiling,
                                       testMatmulToVectorPatterns2dTiling);
  if (testVectorTransferForwardingPatterns)
    return applyVectorTransferForwardingPatterns(getFunction());
  if (testGenericToVectorPattern)
    return applyLinalgToVectorPatterns(getFunction());
  if (testTransformPadTensor)
    return applyPadTensorToGenericPatterns(getFunction());
  if (testGeneralizePadTensor)
    return applyGeneralizePadTensorPatterns(getFunction());
  if (testSwapSubTensorPadTensor)
    return applyExtractSliceOfPadTensorSwapPattern(getFunction());
  if (testTileAndPadPattern)
    return applyTileAndPadPattern(getFunction(), tileSizesForPadding);
  if (testHoistPadding) {
    getFunction().walk([&](linalg::PadTensorOp padTensorOp) {
      (void)linalg::hoistPaddingOnTensors(padTensorOp, testHoistPadding);
    });
  }
  if (testInterchangePattern.hasValue())
    return applyInterchangePattern(getFunction(), testInterchangePattern);
}

namespace mlir {
namespace test {
void registerTestLinalgTransforms() {
  PassRegistration<TestLinalgTransforms>();
}
} // namespace test
} // namespace mlir
