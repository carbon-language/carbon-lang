//===- TosaToArithPass.cpp - Lowering Tosa to Linalg Dialect -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the Arith dialect.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace tosa;

namespace {
struct TosaToArith : public TosaToArithBase<TosaToArith> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addIllegalOp<tosa::ConstOp>();
    target.addLegalDialect<arith::ArithmeticDialect>();

    mlir::tosa::populateTosaToArithConversionPatterns(&patterns);

    if (this->includeApplyRescale) {
      mlir::tosa::populateTosaRescaleToArithConversionPatterns(&patterns,
                                                               this->use32Bit);
      target.addIllegalOp<tosa::ApplyScaleOp>();
    }

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaToArith() {
  return std::make_unique<TosaToArith>();
}
