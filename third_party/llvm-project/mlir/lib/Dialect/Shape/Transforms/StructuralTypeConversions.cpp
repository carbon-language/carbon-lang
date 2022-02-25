//===- StructuralTypeConversions.cpp - Shape structural type conversions --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::shape;

namespace {
class ConvertAssumingOpTypes : public OpConversionPattern<AssumingOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AssumingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type, 2> newResultTypes;
    newResultTypes.reserve(op.getNumResults());
    for (auto result : op.getResults()) {
      auto originalType = result.getType();
      Type convertedType = getTypeConverter()->convertType(originalType);
      newResultTypes.push_back(convertedType);
    }

    auto newAssumingOp = rewriter.create<AssumingOp>(
        op.getLoc(), newResultTypes, op.getWitness());
    rewriter.inlineRegionBefore(op.getDoRegion(), newAssumingOp.getDoRegion(),
                                newAssumingOp.getDoRegion().end());
    rewriter.replaceOp(op, newAssumingOp.getResults());

    return success();
  }
};
} // namespace

namespace {
class ConvertAssumingYieldOpTypes
    : public OpConversionPattern<AssumingYieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AssumingYieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<AssumingYieldOp>(op, adaptor.getOperands());
    return success();
  }
};
} // namespace

void mlir::populateShapeStructuralTypeConversionsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  patterns.add<ConvertAssumingOpTypes, ConvertAssumingYieldOpTypes>(
      typeConverter, patterns.getContext());
  target.addDynamicallyLegalOp<AssumingOp>([&](AssumingOp op) {
    return typeConverter.isLegal(op.getResultTypes());
  });
  target.addDynamicallyLegalOp<AssumingYieldOp>([&](AssumingYieldOp op) {
    return typeConverter.isLegal(op.getOperandTypes());
  });
}
