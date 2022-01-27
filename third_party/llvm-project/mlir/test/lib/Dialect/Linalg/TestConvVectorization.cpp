//===- TestConvVectorization.cpp - Vectorization of Conv ops --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace vector;

namespace {
/// A pass converting MLIR Linalg ops into Vector ops.
class TestConvVectorization
    : public PassWrapper<TestConvVectorization, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "test-conv-vectorization"; }
  StringRef getDescription() const final {
    return "Test vectorization of convolutions";
  }
  TestConvVectorization() = default;
  TestConvVectorization(const TestConvVectorization &) {}
  explicit TestConvVectorization(ArrayRef<int64_t> tileSizesParam) {
    tileSizes = tileSizesParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VectorDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<AffineDialect>();
    registry.insert<StandardOpsDialect>();
  }

  ListOption<int64_t> tileSizes{
      *this, "tile-sizes", llvm::cl::desc("Vectorization sizes."),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};
} // namespace

void TestConvVectorization::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, scf::SCFDialect, StandardOpsDialect,
                         VectorDialect>();
  target.addLegalOp<ModuleOp, FuncOp, ReturnOp>();
  target.addLegalOp<linalg::FillOp, linalg::YieldOp>();

  SmallVector<RewritePatternSet, 4> stage1Patterns;
  linalg::populateConvVectorizationPatterns(context, stage1Patterns, tileSizes);
  SmallVector<FrozenRewritePatternSet, 4> frozenStage1Patterns;
  llvm::move(stage1Patterns, std::back_inserter(frozenStage1Patterns));

  RewritePatternSet stage2Patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  scf::populateSCFForLoopCanonicalizationPatterns(stage2Patterns);

  auto stage3Transforms = [](Operation *op) {
    PassManager pm(op->getContext());
    pm.addPass(createLoopInvariantCodeMotionPass());
    if (failed(pm.run(cast<ModuleOp>(op))))
      llvm_unreachable("Unexpected failure in cleanup pass pipeline.");
    op->walk([](FuncOp func) {
      promoteSingleIterationLoops(func);
      linalg::hoistRedundantVectorTransfers(func);
    });
    return success();
  };

  (void)linalg::applyStagedPatterns(module, frozenStage1Patterns,
                                    std::move(stage2Patterns),
                                    stage3Transforms);

  //===--------------------------------------------------------------------===//
  // Post staged patterns transforms
  //===--------------------------------------------------------------------===//

  VectorTransformsOptions vectorTransformOptions{
      VectorContractLowering::Dot, VectorMultiReductionLowering::InnerParallel,
      VectorTransposeLowering::EltWise};

  RewritePatternSet vectorTransferPatterns(context);
  // Pattern is not applied: rank-reducing vector transfer is not yet supported
  // (see: splitFullAndPartialTransferPrecondition in VectorTransforms.cpp).
  vectorTransferPatterns.add<VectorTransferFullPartialRewriter>(
      context, vectorTransformOptions);
  (void)applyPatternsAndFoldGreedily(module, std::move(vectorTransferPatterns));

  // Programmatic controlled lowering of linalg.copy and linalg.fill.
  PassManager pm(context);
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  if (failed(pm.run(module)))
    llvm_unreachable("Unexpected failure in linalg to loops pass.");

  // Programmatic controlled lowering of vector.contract only.
  RewritePatternSet vectorContractLoweringPatterns(context);
  populateVectorBroadcastLoweringPatterns(vectorContractLoweringPatterns);
  populateVectorContractLoweringPatterns(vectorContractLoweringPatterns,
                                         vectorTransformOptions);
  populateVectorMaskOpLoweringPatterns(vectorContractLoweringPatterns);
  populateVectorShapeCastLoweringPatterns(vectorContractLoweringPatterns);
  populateVectorTransposeLoweringPatterns(vectorContractLoweringPatterns,
                                          vectorTransformOptions);
  (void)applyPatternsAndFoldGreedily(module,
                                     std::move(vectorContractLoweringPatterns));

  // Programmatic controlled lowering of vector.transfer only.
  RewritePatternSet vectorToLoopsPatterns(context);
  populateVectorToSCFConversionPatterns(vectorToLoopsPatterns,
                                        VectorTransferToSCFOptions());
  (void)applyPatternsAndFoldGreedily(module, std::move(vectorToLoopsPatterns));

  // Ensure we drop the marker in the end.
  module.walk([](linalg::LinalgOp op) {
    op->removeAttr(linalg::LinalgTransforms::kLinalgTransformMarker);
  });
}

namespace mlir {
namespace test {
void registerTestConvVectorization() {
  PassRegistration<TestConvVectorization>();
}
} // namespace test
} // namespace mlir
