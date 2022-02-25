//===- MathToLLVM.cpp - Math to LLVM dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace {
using CosOpLowering = VectorConvertToLLVMPattern<math::CosOp, LLVM::CosOp>;
using ExpOpLowering = VectorConvertToLLVMPattern<math::ExpOp, LLVM::ExpOp>;
using Exp2OpLowering = VectorConvertToLLVMPattern<math::Exp2Op, LLVM::Exp2Op>;
using Log10OpLowering =
    VectorConvertToLLVMPattern<math::Log10Op, LLVM::Log10Op>;
using Log2OpLowering = VectorConvertToLLVMPattern<math::Log2Op, LLVM::Log2Op>;
using LogOpLowering = VectorConvertToLLVMPattern<math::LogOp, LLVM::LogOp>;
using PowFOpLowering = VectorConvertToLLVMPattern<math::PowFOp, LLVM::PowOp>;
using SinOpLowering = VectorConvertToLLVMPattern<math::SinOp, LLVM::SinOp>;
using SqrtOpLowering = VectorConvertToLLVMPattern<math::SqrtOp, LLVM::SqrtOp>;

// A `expm1` is converted into `exp - 1`.
struct ExpM1OpLowering : public ConvertOpToLLVMPattern<math::ExpM1Op> {
  using ConvertOpToLLVMPattern<math::ExpM1Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::ExpM1Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    math::ExpM1Op::Adaptor transformed(operands);
    auto operandType = transformed.operand().getType();

    if (!operandType || !LLVM::isCompatibleType(operandType))
      return failure();

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto floatType = getElementTypeOrSelf(resultType).cast<FloatType>();
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);

    if (!operandType.isa<LLVM::LLVMArrayType>()) {
      LLVM::ConstantOp one;
      if (LLVM::isCompatibleVectorType(operandType)) {
        one = rewriter.create<LLVM::ConstantOp>(
            loc, operandType,
            SplatElementsAttr::get(resultType.cast<ShapedType>(), floatOne));
      } else {
        one = rewriter.create<LLVM::ConstantOp>(loc, operandType, floatOne);
      }
      auto exp = rewriter.create<LLVM::ExpOp>(loc, transformed.operand());
      rewriter.replaceOpWithNewOp<LLVM::FSubOp>(op, operandType, exp, one);
      return success();
    }

    auto vectorType = resultType.dyn_cast<VectorType>();
    if (!vectorType)
      return rewriter.notifyMatchFailure(op, "expected vector result type");

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), operands, *getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get(
                  {LLVM::getVectorNumElements(llvm1DVectorTy).getFixedValue()},
                  floatType),
              floatOne);
          auto one =
              rewriter.create<LLVM::ConstantOp>(loc, llvm1DVectorTy, splatAttr);
          auto exp =
              rewriter.create<LLVM::ExpOp>(loc, llvm1DVectorTy, operands[0]);
          return rewriter.create<LLVM::FSubOp>(loc, llvm1DVectorTy, exp, one);
        },
        rewriter);
  }
};

// A `log1p` is converted into `log(1 + ...)`.
struct Log1pOpLowering : public ConvertOpToLLVMPattern<math::Log1pOp> {
  using ConvertOpToLLVMPattern<math::Log1pOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::Log1pOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    math::Log1pOp::Adaptor transformed(operands);
    auto operandType = transformed.operand().getType();

    if (!operandType || !LLVM::isCompatibleType(operandType))
      return rewriter.notifyMatchFailure(op, "unsupported operand type");

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto floatType = getElementTypeOrSelf(resultType).cast<FloatType>();
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);

    if (!operandType.isa<LLVM::LLVMArrayType>()) {
      LLVM::ConstantOp one =
          LLVM::isCompatibleVectorType(operandType)
              ? rewriter.create<LLVM::ConstantOp>(
                    loc, operandType,
                    SplatElementsAttr::get(resultType.cast<ShapedType>(),
                                           floatOne))
              : rewriter.create<LLVM::ConstantOp>(loc, operandType, floatOne);

      auto add = rewriter.create<LLVM::FAddOp>(loc, operandType, one,
                                               transformed.operand());
      rewriter.replaceOpWithNewOp<LLVM::LogOp>(op, operandType, add);
      return success();
    }

    auto vectorType = resultType.dyn_cast<VectorType>();
    if (!vectorType)
      return rewriter.notifyMatchFailure(op, "expected vector result type");

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), operands, *getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get(
                  {LLVM::getVectorNumElements(llvm1DVectorTy).getFixedValue()},
                  floatType),
              floatOne);
          auto one =
              rewriter.create<LLVM::ConstantOp>(loc, llvm1DVectorTy, splatAttr);
          auto add = rewriter.create<LLVM::FAddOp>(loc, llvm1DVectorTy, one,
                                                   operands[0]);
          return rewriter.create<LLVM::LogOp>(loc, llvm1DVectorTy, add);
        },
        rewriter);
  }
};

// A `rsqrt` is converted into `1 / sqrt`.
struct RsqrtOpLowering : public ConvertOpToLLVMPattern<math::RsqrtOp> {
  using ConvertOpToLLVMPattern<math::RsqrtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::RsqrtOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    math::RsqrtOp::Adaptor transformed(operands);
    auto operandType = transformed.operand().getType();

    if (!operandType || !LLVM::isCompatibleType(operandType))
      return failure();

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto floatType = getElementTypeOrSelf(resultType).cast<FloatType>();
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);

    if (!operandType.isa<LLVM::LLVMArrayType>()) {
      LLVM::ConstantOp one;
      if (LLVM::isCompatibleVectorType(operandType)) {
        one = rewriter.create<LLVM::ConstantOp>(
            loc, operandType,
            SplatElementsAttr::get(resultType.cast<ShapedType>(), floatOne));
      } else {
        one = rewriter.create<LLVM::ConstantOp>(loc, operandType, floatOne);
      }
      auto sqrt = rewriter.create<LLVM::SqrtOp>(loc, transformed.operand());
      rewriter.replaceOpWithNewOp<LLVM::FDivOp>(op, operandType, one, sqrt);
      return success();
    }

    auto vectorType = resultType.dyn_cast<VectorType>();
    if (!vectorType)
      return failure();

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), operands, *getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get(
                  {LLVM::getVectorNumElements(llvm1DVectorTy).getFixedValue()},
                  floatType),
              floatOne);
          auto one =
              rewriter.create<LLVM::ConstantOp>(loc, llvm1DVectorTy, splatAttr);
          auto sqrt =
              rewriter.create<LLVM::SqrtOp>(loc, llvm1DVectorTy, operands[0]);
          return rewriter.create<LLVM::FDivOp>(loc, llvm1DVectorTy, one, sqrt);
        },
        rewriter);
  }
};

struct ConvertMathToLLVMPass
    : public ConvertMathToLLVMBase<ConvertMathToLLVMPass> {
  ConvertMathToLLVMPass() = default;

  void runOnFunction() override {
    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext());
    populateMathToLLVMConversionPatterns(converter, patterns);
    LLVMConversionTarget target(getContext());
    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::populateMathToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    CosOpLowering,
    ExpOpLowering,
    Exp2OpLowering,
    ExpM1OpLowering,
    Log10OpLowering,
    Log1pOpLowering,
    Log2OpLowering,
    LogOpLowering,
    PowFOpLowering,
    RsqrtOpLowering,
    SinOpLowering,
    SqrtOpLowering
  >(converter);
  // clang-format on
}

std::unique_ptr<Pass> mlir::createConvertMathToLLVMPass() {
  return std::make_unique<ConvertMathToLLVMPass>();
}
