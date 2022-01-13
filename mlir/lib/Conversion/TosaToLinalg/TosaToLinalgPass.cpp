//===- TosaToLinalgPass.cpp - Lowering Tosa to Linalg Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TosaToLinalgOnTensors
    : public TosaToLinalgOnTensorsBase<TosaToLinalgOnTensors> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, math::MathDialect,
                    StandardOpsDialect, tensor::TensorDialect>();
  }

  void runOnFunction() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           tensor::TensorDialect>();
    target.addIllegalDialect<tosa::TosaDialect>();

    // Not every TOSA op can be legalized to linalg.
    target.addLegalOp<tosa::ApplyScaleOp>();
    target.addLegalOp<tosa::IfOp>();
    target.addLegalOp<tosa::ConstOp>();
    target.addLegalOp<tosa::WhileOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    FuncOp func = getFunction();
    mlir::tosa::populateTosaToLinalgOnTensorsConversionPatterns(&patterns);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaToLinalgOnTensors() {
  return std::make_unique<TosaToLinalgOnTensors>();
}

void mlir::tosa::addTosaToLinalgOnTensorsPasses(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(createTosaMakeBroadcastablePass());
  pm.addNestedPass<FuncOp>(createTosaToLinalgOnTensors());
}
