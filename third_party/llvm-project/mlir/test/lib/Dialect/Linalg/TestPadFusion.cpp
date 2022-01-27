//===- TestPadFusion.cpp - Test fusion of pad op with Linalg ops ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing fusion of pad ops with its producer
// Linalg op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

namespace {
struct TestPadFusionPass : public PassWrapper<TestPadFusionPass, FunctionPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, linalg::LinalgDialect, tensor::TensorDialect>();
  }

  StringRef getArgument() const final { return "test-linalg-pad-fusion"; }
  StringRef getDescription() const final { return "Test PadOp fusion"; }

  void runOnFunction() override {
    MLIRContext *context = &getContext();
    FuncOp funcOp = getFunction();
    RewritePatternSet patterns(context);
    linalg::populateFusePadTensorWithProducerLinalgOpPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp.getBody(),
                                            std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

namespace test {
void registerTestPadFusion() { PassRegistration<TestPadFusionPass>(); }
} // namespace test

} // namespace mlir
