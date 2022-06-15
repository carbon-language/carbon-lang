//===- MathToSPIRV.cpp - Math to SPIR-V Patterns --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Math dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "../SPIRVCommon/Pattern.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "math-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Creates a 32-bit scalar/vector integer constant. Returns nullptr if the
/// given type is not a 32-bit scalar/vector type.
static Value getScalarOrVectorI32Constant(Type type, int value,
                                          OpBuilder &builder, Location loc) {
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    if (!vectorType.getElementType().isInteger(32))
      return nullptr;
    SmallVector<int> values(vectorType.getNumElements(), value);
    return builder.create<spirv::ConstantOp>(loc, type,
                                             builder.getI32VectorAttr(values));
  }
  if (type.isInteger(32))
    return builder.create<spirv::ConstantOp>(loc, type,
                                             builder.getI32IntegerAttr(value));

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

// Note that DRR cannot be used for the patterns in this file: we may need to
// convert type along the way, which requires ConversionPattern. DRR generates
// normal RewritePattern.

namespace {
/// Converts math.copysign to SPIR-V ops.
class CopySignPattern final : public OpConversionPattern<math::CopySignOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::CopySignOp copySignOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = getTypeConverter()->convertType(copySignOp.getType());
    if (!type)
      return failure();

    FloatType floatType;
    if (auto scalarType = copySignOp.getType().dyn_cast<FloatType>()) {
      floatType = scalarType;
    } else if (auto vectorType = copySignOp.getType().dyn_cast<VectorType>()) {
      floatType = vectorType.getElementType().cast<FloatType>();
    } else {
      return failure();
    }

    Location loc = copySignOp.getLoc();
    int bitwidth = floatType.getWidth();
    Type intType = rewriter.getIntegerType(bitwidth);
    uint64_t intValue = uint64_t(1) << (bitwidth - 1);

    Value signMask = rewriter.create<spirv::ConstantOp>(
        loc, intType, rewriter.getIntegerAttr(intType, intValue));
    Value valueMask = rewriter.create<spirv::ConstantOp>(
        loc, intType, rewriter.getIntegerAttr(intType, intValue - 1u));

    if (auto vectorType = copySignOp.getType().dyn_cast<VectorType>()) {
      assert(vectorType.getRank() == 1);
      int count = vectorType.getNumElements();
      intType = VectorType::get(count, intType);

      SmallVector<Value> signSplat(count, signMask);
      signMask =
          rewriter.create<spirv::CompositeConstructOp>(loc, intType, signSplat);

      SmallVector<Value> valueSplat(count, valueMask);
      valueMask = rewriter.create<spirv::CompositeConstructOp>(loc, intType,
                                                               valueSplat);
    }

    Value lhsCast =
        rewriter.create<spirv::BitcastOp>(loc, intType, adaptor.getLhs());
    Value rhsCast =
        rewriter.create<spirv::BitcastOp>(loc, intType, adaptor.getRhs());

    Value value = rewriter.create<spirv::BitwiseAndOp>(
        loc, intType, ValueRange{lhsCast, valueMask});
    Value sign = rewriter.create<spirv::BitwiseAndOp>(
        loc, intType, ValueRange{rhsCast, signMask});

    Value result = rewriter.create<spirv::BitwiseOrOp>(loc, intType,
                                                       ValueRange{value, sign});
    rewriter.replaceOpWithNewOp<spirv::BitcastOp>(copySignOp, type, result);
    return success();
  }
};

/// Converts math.ctlz to SPIR-V ops.
///
/// SPIR-V does not have a direct operations for counting leading zeros. If
/// Shader capability is supported, we can leverage GLSL FindUMsb to calculate
/// it.
class CountLeadingZerosPattern final
    : public OpConversionPattern<math::CountLeadingZerosOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::CountLeadingZerosOp countOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = getTypeConverter()->convertType(countOp.getType());
    if (!type)
      return failure();

    // We can only support 32-bit integer types for now.
    unsigned bitwidth = 0;
    if (type.isa<IntegerType>())
      bitwidth = type.getIntOrFloatBitWidth();
    if (auto vectorType = type.dyn_cast<VectorType>())
      bitwidth = vectorType.getElementTypeBitWidth();
    if (bitwidth != 32)
      return failure();

    Location loc = countOp.getLoc();
    Value val31 = getScalarOrVectorI32Constant(type, 31, rewriter, loc);
    Value msb =
        rewriter.create<spirv::GLSLFindUMsbOp>(loc, adaptor.getOperand());
    // We need to subtract from 31 given that the index is from the least
    // significant bit.
    rewriter.replaceOpWithNewOp<spirv::ISubOp>(countOp, val31, msb);
    return success();
  }
};

/// Converts math.expm1 to SPIR-V ops.
///
/// SPIR-V does not have a direct operations for exp(x)-1. Explicitly lower to
/// these operations.
template <typename ExpOp>
struct ExpM1OpPattern final : public OpConversionPattern<math::ExpM1Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ExpM1Op operation, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(adaptor.getOperands().size() == 1);
    Location loc = operation.getLoc();
    auto type = this->getTypeConverter()->convertType(operation.getType());
    auto exp = rewriter.create<ExpOp>(loc, type, adaptor.getOperand());
    auto one = spirv::ConstantOp::getOne(type, loc, rewriter);
    rewriter.replaceOpWithNewOp<spirv::FSubOp>(operation, exp, one);
    return success();
  }
};

/// Converts math.log1p to SPIR-V ops.
///
/// SPIR-V does not have a direct operations for log(1+x). Explicitly lower to
/// these operations.
template <typename LogOp>
struct Log1pOpPattern final : public OpConversionPattern<math::Log1pOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::Log1pOp operation, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(adaptor.getOperands().size() == 1);
    Location loc = operation.getLoc();
    auto type = this->getTypeConverter()->convertType(operation.getType());
    auto one = spirv::ConstantOp::getOne(type, operation.getLoc(), rewriter);
    auto onePlus =
        rewriter.create<spirv::FAddOp>(loc, one, adaptor.getOperand());
    rewriter.replaceOpWithNewOp<LogOp>(operation, type, onePlus);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

namespace mlir {
void populateMathToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                 RewritePatternSet &patterns) {
  // Core patterns
  patterns.add<CopySignPattern>(typeConverter, patterns.getContext());

  // GLSL patterns
  patterns
      .add<CountLeadingZerosPattern, Log1pOpPattern<spirv::GLSLLogOp>,
           ExpM1OpPattern<spirv::GLSLExpOp>,
           spirv::ElementwiseOpPattern<math::AbsOp, spirv::GLSLFAbsOp>,
           spirv::ElementwiseOpPattern<math::CeilOp, spirv::GLSLCeilOp>,
           spirv::ElementwiseOpPattern<math::CosOp, spirv::GLSLCosOp>,
           spirv::ElementwiseOpPattern<math::ExpOp, spirv::GLSLExpOp>,
           spirv::ElementwiseOpPattern<math::FloorOp, spirv::GLSLFloorOp>,
           spirv::ElementwiseOpPattern<math::FmaOp, spirv::GLSLFmaOp>,
           spirv::ElementwiseOpPattern<math::LogOp, spirv::GLSLLogOp>,
           spirv::ElementwiseOpPattern<math::PowFOp, spirv::GLSLPowOp>,
           spirv::ElementwiseOpPattern<math::RsqrtOp, spirv::GLSLInverseSqrtOp>,
           spirv::ElementwiseOpPattern<math::SinOp, spirv::GLSLSinOp>,
           spirv::ElementwiseOpPattern<math::SqrtOp, spirv::GLSLSqrtOp>,
           spirv::ElementwiseOpPattern<math::TanhOp, spirv::GLSLTanhOp>>(
          typeConverter, patterns.getContext());

  // OpenCL patterns
  patterns.add<Log1pOpPattern<spirv::OCLLogOp>, ExpM1OpPattern<spirv::OCLExpOp>,
               spirv::ElementwiseOpPattern<math::AbsOp, spirv::OCLFAbsOp>,
               spirv::ElementwiseOpPattern<math::CeilOp, spirv::OCLCeilOp>,
               spirv::ElementwiseOpPattern<math::CosOp, spirv::OCLCosOp>,
               spirv::ElementwiseOpPattern<math::ErfOp, spirv::OCLErfOp>,
               spirv::ElementwiseOpPattern<math::ExpOp, spirv::OCLExpOp>,
               spirv::ElementwiseOpPattern<math::FloorOp, spirv::OCLFloorOp>,
               spirv::ElementwiseOpPattern<math::FmaOp, spirv::OCLFmaOp>,
               spirv::ElementwiseOpPattern<math::LogOp, spirv::OCLLogOp>,
               spirv::ElementwiseOpPattern<math::PowFOp, spirv::OCLPowOp>,
               spirv::ElementwiseOpPattern<math::RsqrtOp, spirv::OCLRsqrtOp>,
               spirv::ElementwiseOpPattern<math::SinOp, spirv::OCLSinOp>,
               spirv::ElementwiseOpPattern<math::SqrtOp, spirv::OCLSqrtOp>,
               spirv::ElementwiseOpPattern<math::TanhOp, spirv::OCLTanhOp>>(
      typeConverter, patterns.getContext());
}

} // namespace mlir
