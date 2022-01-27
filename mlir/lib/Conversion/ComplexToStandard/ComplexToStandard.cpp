//===- ComplexToStandard.cpp - conversion from Complex to Standard dialect ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"

#include <memory>
#include <type_traits>

#include "../PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct AbsOpConversion : public OpConversionPattern<complex::AbsOp> {
  using OpConversionPattern<complex::AbsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = op.getType();

    Value real =
        rewriter.create<complex::ReOp>(loc, type, adaptor.getComplex());
    Value imag =
        rewriter.create<complex::ImOp>(loc, type, adaptor.getComplex());
    Value realSqr = rewriter.create<arith::MulFOp>(loc, real, real);
    Value imagSqr = rewriter.create<arith::MulFOp>(loc, imag, imag);
    Value sqNorm = rewriter.create<arith::AddFOp>(loc, realSqr, imagSqr);

    rewriter.replaceOpWithNewOp<math::SqrtOp>(op, sqNorm);
    return success();
  }
};

template <typename ComparisonOp, arith::CmpFPredicate p>
struct ComparisonOpConversion : public OpConversionPattern<ComparisonOp> {
  using OpConversionPattern<ComparisonOp>::OpConversionPattern;
  using ResultCombiner =
      std::conditional_t<std::is_same<ComparisonOp, complex::EqualOp>::value,
                         arith::AndIOp, arith::OrIOp>;

  LogicalResult
  matchAndRewrite(ComparisonOp op, typename ComparisonOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = adaptor.getLhs()
                    .getType()
                    .template cast<ComplexType>()
                    .getElementType();

    Value realLhs = rewriter.create<complex::ReOp>(loc, type, adaptor.getLhs());
    Value imagLhs = rewriter.create<complex::ImOp>(loc, type, adaptor.getLhs());
    Value realRhs = rewriter.create<complex::ReOp>(loc, type, adaptor.getRhs());
    Value imagRhs = rewriter.create<complex::ImOp>(loc, type, adaptor.getRhs());
    Value realComparison =
        rewriter.create<arith::CmpFOp>(loc, p, realLhs, realRhs);
    Value imagComparison =
        rewriter.create<arith::CmpFOp>(loc, p, imagLhs, imagRhs);

    rewriter.replaceOpWithNewOp<ResultCombiner>(op, realComparison,
                                                imagComparison);
    return success();
  }
};

// Default conversion which applies the BinaryStandardOp separately on the real
// and imaginary parts. Can for example be used for complex::AddOp and
// complex::SubOp.
template <typename BinaryComplexOp, typename BinaryStandardOp>
struct BinaryComplexOpConversion : public OpConversionPattern<BinaryComplexOp> {
  using OpConversionPattern<BinaryComplexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BinaryComplexOp op, typename BinaryComplexOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = adaptor.getLhs().getType().template cast<ComplexType>();
    auto elementType = type.getElementType().template cast<FloatType>();
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value realLhs = b.create<complex::ReOp>(elementType, adaptor.getLhs());
    Value realRhs = b.create<complex::ReOp>(elementType, adaptor.getRhs());
    Value resultReal =
        b.create<BinaryStandardOp>(elementType, realLhs, realRhs);
    Value imagLhs = b.create<complex::ImOp>(elementType, adaptor.getLhs());
    Value imagRhs = b.create<complex::ImOp>(elementType, adaptor.getRhs());
    Value resultImag =
        b.create<BinaryStandardOp>(elementType, imagLhs, imagRhs);
    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultReal,
                                                   resultImag);
    return success();
  }
};

struct DivOpConversion : public OpConversionPattern<complex::DivOp> {
  using OpConversionPattern<complex::DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = adaptor.getLhs().getType().cast<ComplexType>();
    auto elementType = type.getElementType().cast<FloatType>();

    Value lhsReal =
        rewriter.create<complex::ReOp>(loc, elementType, adaptor.getLhs());
    Value lhsImag =
        rewriter.create<complex::ImOp>(loc, elementType, adaptor.getLhs());
    Value rhsReal =
        rewriter.create<complex::ReOp>(loc, elementType, adaptor.getRhs());
    Value rhsImag =
        rewriter.create<complex::ImOp>(loc, elementType, adaptor.getRhs());

    // Smith's algorithm to divide complex numbers. It is just a bit smarter
    // way to compute the following formula:
    //  (lhsReal + lhsImag * i) / (rhsReal + rhsImag * i)
    //    = (lhsReal + lhsImag * i) (rhsReal - rhsImag * i) /
    //          ((rhsReal + rhsImag * i)(rhsReal - rhsImag * i))
    //    = ((lhsReal * rhsReal + lhsImag * rhsImag) +
    //          (lhsImag * rhsReal - lhsReal * rhsImag) * i) / ||rhs||^2
    //
    // Depending on whether |rhsReal| < |rhsImag| we compute either
    //   rhsRealImagRatio = rhsReal / rhsImag
    //   rhsRealImagDenom = rhsImag + rhsReal * rhsRealImagRatio
    //   resultReal = (lhsReal * rhsRealImagRatio + lhsImag) / rhsRealImagDenom
    //   resultImag = (lhsImag * rhsRealImagRatio - lhsReal) / rhsRealImagDenom
    //
    // or
    //
    //   rhsImagRealRatio = rhsImag / rhsReal
    //   rhsImagRealDenom = rhsReal + rhsImag * rhsImagRealRatio
    //   resultReal = (lhsReal + lhsImag * rhsImagRealRatio) / rhsImagRealDenom
    //   resultImag = (lhsImag - lhsReal * rhsImagRealRatio) / rhsImagRealDenom
    //
    // See https://dl.acm.org/citation.cfm?id=368661 for more details.
    Value rhsRealImagRatio =
        rewriter.create<arith::DivFOp>(loc, rhsReal, rhsImag);
    Value rhsRealImagDenom = rewriter.create<arith::AddFOp>(
        loc, rhsImag,
        rewriter.create<arith::MulFOp>(loc, rhsRealImagRatio, rhsReal));
    Value realNumerator1 = rewriter.create<arith::AddFOp>(
        loc, rewriter.create<arith::MulFOp>(loc, lhsReal, rhsRealImagRatio),
        lhsImag);
    Value resultReal1 =
        rewriter.create<arith::DivFOp>(loc, realNumerator1, rhsRealImagDenom);
    Value imagNumerator1 = rewriter.create<arith::SubFOp>(
        loc, rewriter.create<arith::MulFOp>(loc, lhsImag, rhsRealImagRatio),
        lhsReal);
    Value resultImag1 =
        rewriter.create<arith::DivFOp>(loc, imagNumerator1, rhsRealImagDenom);

    Value rhsImagRealRatio =
        rewriter.create<arith::DivFOp>(loc, rhsImag, rhsReal);
    Value rhsImagRealDenom = rewriter.create<arith::AddFOp>(
        loc, rhsReal,
        rewriter.create<arith::MulFOp>(loc, rhsImagRealRatio, rhsImag));
    Value realNumerator2 = rewriter.create<arith::AddFOp>(
        loc, lhsReal,
        rewriter.create<arith::MulFOp>(loc, lhsImag, rhsImagRealRatio));
    Value resultReal2 =
        rewriter.create<arith::DivFOp>(loc, realNumerator2, rhsImagRealDenom);
    Value imagNumerator2 = rewriter.create<arith::SubFOp>(
        loc, lhsImag,
        rewriter.create<arith::MulFOp>(loc, lhsReal, rhsImagRealRatio));
    Value resultImag2 =
        rewriter.create<arith::DivFOp>(loc, imagNumerator2, rhsImagRealDenom);

    // Consider corner cases.
    // Case 1. Zero denominator, numerator contains at most one NaN value.
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getZeroAttr(elementType));
    Value rhsRealAbs = rewriter.create<math::AbsOp>(loc, rhsReal);
    Value rhsRealIsZero = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, rhsRealAbs, zero);
    Value rhsImagAbs = rewriter.create<math::AbsOp>(loc, rhsImag);
    Value rhsImagIsZero = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, rhsImagAbs, zero);
    Value lhsRealIsNotNaN = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ORD, lhsReal, zero);
    Value lhsImagIsNotNaN = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ORD, lhsImag, zero);
    Value lhsContainsNotNaNValue =
        rewriter.create<arith::OrIOp>(loc, lhsRealIsNotNaN, lhsImagIsNotNaN);
    Value resultIsInfinity = rewriter.create<arith::AndIOp>(
        loc, lhsContainsNotNaNValue,
        rewriter.create<arith::AndIOp>(loc, rhsRealIsZero, rhsImagIsZero));
    Value inf = rewriter.create<arith::ConstantOp>(
        loc, elementType,
        rewriter.getFloatAttr(
            elementType, APFloat::getInf(elementType.getFloatSemantics())));
    Value infWithSignOfRhsReal =
        rewriter.create<math::CopySignOp>(loc, inf, rhsReal);
    Value infinityResultReal =
        rewriter.create<arith::MulFOp>(loc, infWithSignOfRhsReal, lhsReal);
    Value infinityResultImag =
        rewriter.create<arith::MulFOp>(loc, infWithSignOfRhsReal, lhsImag);

    // Case 2. Infinite numerator, finite denominator.
    Value rhsRealFinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, rhsRealAbs, inf);
    Value rhsImagFinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, rhsImagAbs, inf);
    Value rhsFinite =
        rewriter.create<arith::AndIOp>(loc, rhsRealFinite, rhsImagFinite);
    Value lhsRealAbs = rewriter.create<math::AbsOp>(loc, lhsReal);
    Value lhsRealInfinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, lhsRealAbs, inf);
    Value lhsImagAbs = rewriter.create<math::AbsOp>(loc, lhsImag);
    Value lhsImagInfinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, lhsImagAbs, inf);
    Value lhsInfinite =
        rewriter.create<arith::OrIOp>(loc, lhsRealInfinite, lhsImagInfinite);
    Value infNumFiniteDenom =
        rewriter.create<arith::AndIOp>(loc, lhsInfinite, rhsFinite);
    Value one = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 1));
    Value lhsRealIsInfWithSign = rewriter.create<math::CopySignOp>(
        loc, rewriter.create<SelectOp>(loc, lhsRealInfinite, one, zero),
        lhsReal);
    Value lhsImagIsInfWithSign = rewriter.create<math::CopySignOp>(
        loc, rewriter.create<SelectOp>(loc, lhsImagInfinite, one, zero),
        lhsImag);
    Value lhsRealIsInfWithSignTimesRhsReal =
        rewriter.create<arith::MulFOp>(loc, lhsRealIsInfWithSign, rhsReal);
    Value lhsImagIsInfWithSignTimesRhsImag =
        rewriter.create<arith::MulFOp>(loc, lhsImagIsInfWithSign, rhsImag);
    Value resultReal3 = rewriter.create<arith::MulFOp>(
        loc, inf,
        rewriter.create<arith::AddFOp>(loc, lhsRealIsInfWithSignTimesRhsReal,
                                       lhsImagIsInfWithSignTimesRhsImag));
    Value lhsRealIsInfWithSignTimesRhsImag =
        rewriter.create<arith::MulFOp>(loc, lhsRealIsInfWithSign, rhsImag);
    Value lhsImagIsInfWithSignTimesRhsReal =
        rewriter.create<arith::MulFOp>(loc, lhsImagIsInfWithSign, rhsReal);
    Value resultImag3 = rewriter.create<arith::MulFOp>(
        loc, inf,
        rewriter.create<arith::SubFOp>(loc, lhsImagIsInfWithSignTimesRhsReal,
                                       lhsRealIsInfWithSignTimesRhsImag));

    // Case 3: Finite numerator, infinite denominator.
    Value lhsRealFinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, lhsRealAbs, inf);
    Value lhsImagFinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, lhsImagAbs, inf);
    Value lhsFinite =
        rewriter.create<arith::AndIOp>(loc, lhsRealFinite, lhsImagFinite);
    Value rhsRealInfinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, rhsRealAbs, inf);
    Value rhsImagInfinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, rhsImagAbs, inf);
    Value rhsInfinite =
        rewriter.create<arith::OrIOp>(loc, rhsRealInfinite, rhsImagInfinite);
    Value finiteNumInfiniteDenom =
        rewriter.create<arith::AndIOp>(loc, lhsFinite, rhsInfinite);
    Value rhsRealIsInfWithSign = rewriter.create<math::CopySignOp>(
        loc, rewriter.create<SelectOp>(loc, rhsRealInfinite, one, zero),
        rhsReal);
    Value rhsImagIsInfWithSign = rewriter.create<math::CopySignOp>(
        loc, rewriter.create<SelectOp>(loc, rhsImagInfinite, one, zero),
        rhsImag);
    Value rhsRealIsInfWithSignTimesLhsReal =
        rewriter.create<arith::MulFOp>(loc, lhsReal, rhsRealIsInfWithSign);
    Value rhsImagIsInfWithSignTimesLhsImag =
        rewriter.create<arith::MulFOp>(loc, lhsImag, rhsImagIsInfWithSign);
    Value resultReal4 = rewriter.create<arith::MulFOp>(
        loc, zero,
        rewriter.create<arith::AddFOp>(loc, rhsRealIsInfWithSignTimesLhsReal,
                                       rhsImagIsInfWithSignTimesLhsImag));
    Value rhsRealIsInfWithSignTimesLhsImag =
        rewriter.create<arith::MulFOp>(loc, lhsImag, rhsRealIsInfWithSign);
    Value rhsImagIsInfWithSignTimesLhsReal =
        rewriter.create<arith::MulFOp>(loc, lhsReal, rhsImagIsInfWithSign);
    Value resultImag4 = rewriter.create<arith::MulFOp>(
        loc, zero,
        rewriter.create<arith::SubFOp>(loc, rhsRealIsInfWithSignTimesLhsImag,
                                       rhsImagIsInfWithSignTimesLhsReal));

    Value realAbsSmallerThanImagAbs = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, rhsRealAbs, rhsImagAbs);
    Value resultReal = rewriter.create<SelectOp>(loc, realAbsSmallerThanImagAbs,
                                                 resultReal1, resultReal2);
    Value resultImag = rewriter.create<SelectOp>(loc, realAbsSmallerThanImagAbs,
                                                 resultImag1, resultImag2);
    Value resultRealSpecialCase3 = rewriter.create<SelectOp>(
        loc, finiteNumInfiniteDenom, resultReal4, resultReal);
    Value resultImagSpecialCase3 = rewriter.create<SelectOp>(
        loc, finiteNumInfiniteDenom, resultImag4, resultImag);
    Value resultRealSpecialCase2 = rewriter.create<SelectOp>(
        loc, infNumFiniteDenom, resultReal3, resultRealSpecialCase3);
    Value resultImagSpecialCase2 = rewriter.create<SelectOp>(
        loc, infNumFiniteDenom, resultImag3, resultImagSpecialCase3);
    Value resultRealSpecialCase1 = rewriter.create<SelectOp>(
        loc, resultIsInfinity, infinityResultReal, resultRealSpecialCase2);
    Value resultImagSpecialCase1 = rewriter.create<SelectOp>(
        loc, resultIsInfinity, infinityResultImag, resultImagSpecialCase2);

    Value resultRealIsNaN = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::UNO, resultReal, zero);
    Value resultImagIsNaN = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::UNO, resultImag, zero);
    Value resultIsNaN =
        rewriter.create<arith::AndIOp>(loc, resultRealIsNaN, resultImagIsNaN);
    Value resultRealWithSpecialCases = rewriter.create<SelectOp>(
        loc, resultIsNaN, resultRealSpecialCase1, resultReal);
    Value resultImagWithSpecialCases = rewriter.create<SelectOp>(
        loc, resultIsNaN, resultImagSpecialCase1, resultImag);

    rewriter.replaceOpWithNewOp<complex::CreateOp>(
        op, type, resultRealWithSpecialCases, resultImagWithSpecialCases);
    return success();
  }
};

struct ExpOpConversion : public OpConversionPattern<complex::ExpOp> {
  using OpConversionPattern<complex::ExpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::ExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = adaptor.getComplex().getType().cast<ComplexType>();
    auto elementType = type.getElementType().cast<FloatType>();

    Value real =
        rewriter.create<complex::ReOp>(loc, elementType, adaptor.getComplex());
    Value imag =
        rewriter.create<complex::ImOp>(loc, elementType, adaptor.getComplex());
    Value expReal = rewriter.create<math::ExpOp>(loc, real);
    Value cosImag = rewriter.create<math::CosOp>(loc, imag);
    Value resultReal = rewriter.create<arith::MulFOp>(loc, expReal, cosImag);
    Value sinImag = rewriter.create<math::SinOp>(loc, imag);
    Value resultImag = rewriter.create<arith::MulFOp>(loc, expReal, sinImag);

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultReal,
                                                   resultImag);
    return success();
  }
};

struct LogOpConversion : public OpConversionPattern<complex::LogOp> {
  using OpConversionPattern<complex::LogOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::LogOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = adaptor.getComplex().getType().cast<ComplexType>();
    auto elementType = type.getElementType().cast<FloatType>();
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value abs = b.create<complex::AbsOp>(elementType, adaptor.getComplex());
    Value resultReal = b.create<math::LogOp>(elementType, abs);
    Value real = b.create<complex::ReOp>(elementType, adaptor.getComplex());
    Value imag = b.create<complex::ImOp>(elementType, adaptor.getComplex());
    Value resultImag = b.create<math::Atan2Op>(elementType, imag, real);
    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultReal,
                                                   resultImag);
    return success();
  }
};

struct Log1pOpConversion : public OpConversionPattern<complex::Log1pOp> {
  using OpConversionPattern<complex::Log1pOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::Log1pOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = adaptor.getComplex().getType().cast<ComplexType>();
    auto elementType = type.getElementType().cast<FloatType>();
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value real = b.create<complex::ReOp>(elementType, adaptor.getComplex());
    Value imag = b.create<complex::ImOp>(elementType, adaptor.getComplex());
    Value one = b.create<arith::ConstantOp>(elementType,
                                            b.getFloatAttr(elementType, 1));
    Value realPlusOne = b.create<arith::AddFOp>(real, one);
    Value newComplex = b.create<complex::CreateOp>(type, realPlusOne, imag);
    rewriter.replaceOpWithNewOp<complex::LogOp>(op, type, newComplex);
    return success();
  }
};

struct MulOpConversion : public OpConversionPattern<complex::MulOp> {
  using OpConversionPattern<complex::MulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto type = adaptor.getLhs().getType().cast<ComplexType>();
    auto elementType = type.getElementType().cast<FloatType>();

    Value lhsReal = b.create<complex::ReOp>(elementType, adaptor.getLhs());
    Value lhsRealAbs = b.create<math::AbsOp>(lhsReal);
    Value lhsImag = b.create<complex::ImOp>(elementType, adaptor.getLhs());
    Value lhsImagAbs = b.create<math::AbsOp>(lhsImag);
    Value rhsReal = b.create<complex::ReOp>(elementType, adaptor.getRhs());
    Value rhsRealAbs = b.create<math::AbsOp>(rhsReal);
    Value rhsImag = b.create<complex::ImOp>(elementType, adaptor.getRhs());
    Value rhsImagAbs = b.create<math::AbsOp>(rhsImag);

    Value lhsRealTimesRhsReal = b.create<arith::MulFOp>(lhsReal, rhsReal);
    Value lhsRealTimesRhsRealAbs = b.create<math::AbsOp>(lhsRealTimesRhsReal);
    Value lhsImagTimesRhsImag = b.create<arith::MulFOp>(lhsImag, rhsImag);
    Value lhsImagTimesRhsImagAbs = b.create<math::AbsOp>(lhsImagTimesRhsImag);
    Value real =
        b.create<arith::SubFOp>(lhsRealTimesRhsReal, lhsImagTimesRhsImag);

    Value lhsImagTimesRhsReal = b.create<arith::MulFOp>(lhsImag, rhsReal);
    Value lhsImagTimesRhsRealAbs = b.create<math::AbsOp>(lhsImagTimesRhsReal);
    Value lhsRealTimesRhsImag = b.create<arith::MulFOp>(lhsReal, rhsImag);
    Value lhsRealTimesRhsImagAbs = b.create<math::AbsOp>(lhsRealTimesRhsImag);
    Value imag =
        b.create<arith::AddFOp>(lhsImagTimesRhsReal, lhsRealTimesRhsImag);

    // Handle cases where the "naive" calculation results in NaN values.
    Value realIsNan =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, real, real);
    Value imagIsNan =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, imag, imag);
    Value isNan = b.create<arith::AndIOp>(realIsNan, imagIsNan);

    Value inf = b.create<arith::ConstantOp>(
        elementType,
        b.getFloatAttr(elementType,
                       APFloat::getInf(elementType.getFloatSemantics())));

    // Case 1. `lhsReal` or `lhsImag` are infinite.
    Value lhsRealIsInf =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, lhsRealAbs, inf);
    Value lhsImagIsInf =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, lhsImagAbs, inf);
    Value lhsIsInf = b.create<arith::OrIOp>(lhsRealIsInf, lhsImagIsInf);
    Value rhsRealIsNan =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, rhsReal, rhsReal);
    Value rhsImagIsNan =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, rhsImag, rhsImag);
    Value zero =
        b.create<arith::ConstantOp>(elementType, b.getZeroAttr(elementType));
    Value one = b.create<arith::ConstantOp>(elementType,
                                            b.getFloatAttr(elementType, 1));
    Value lhsRealIsInfFloat = b.create<SelectOp>(lhsRealIsInf, one, zero);
    lhsReal = b.create<SelectOp>(
        lhsIsInf, b.create<math::CopySignOp>(lhsRealIsInfFloat, lhsReal),
        lhsReal);
    Value lhsImagIsInfFloat = b.create<SelectOp>(lhsImagIsInf, one, zero);
    lhsImag = b.create<SelectOp>(
        lhsIsInf, b.create<math::CopySignOp>(lhsImagIsInfFloat, lhsImag),
        lhsImag);
    Value lhsIsInfAndRhsRealIsNan =
        b.create<arith::AndIOp>(lhsIsInf, rhsRealIsNan);
    rhsReal =
        b.create<SelectOp>(lhsIsInfAndRhsRealIsNan,
                           b.create<math::CopySignOp>(zero, rhsReal), rhsReal);
    Value lhsIsInfAndRhsImagIsNan =
        b.create<arith::AndIOp>(lhsIsInf, rhsImagIsNan);
    rhsImag =
        b.create<SelectOp>(lhsIsInfAndRhsImagIsNan,
                           b.create<math::CopySignOp>(zero, rhsImag), rhsImag);

    // Case 2. `rhsReal` or `rhsImag` are infinite.
    Value rhsRealIsInf =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, rhsRealAbs, inf);
    Value rhsImagIsInf =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, rhsImagAbs, inf);
    Value rhsIsInf = b.create<arith::OrIOp>(rhsRealIsInf, rhsImagIsInf);
    Value lhsRealIsNan =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, lhsReal, lhsReal);
    Value lhsImagIsNan =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, lhsImag, lhsImag);
    Value rhsRealIsInfFloat = b.create<SelectOp>(rhsRealIsInf, one, zero);
    rhsReal = b.create<SelectOp>(
        rhsIsInf, b.create<math::CopySignOp>(rhsRealIsInfFloat, rhsReal),
        rhsReal);
    Value rhsImagIsInfFloat = b.create<SelectOp>(rhsImagIsInf, one, zero);
    rhsImag = b.create<SelectOp>(
        rhsIsInf, b.create<math::CopySignOp>(rhsImagIsInfFloat, rhsImag),
        rhsImag);
    Value rhsIsInfAndLhsRealIsNan =
        b.create<arith::AndIOp>(rhsIsInf, lhsRealIsNan);
    lhsReal =
        b.create<SelectOp>(rhsIsInfAndLhsRealIsNan,
                           b.create<math::CopySignOp>(zero, lhsReal), lhsReal);
    Value rhsIsInfAndLhsImagIsNan =
        b.create<arith::AndIOp>(rhsIsInf, lhsImagIsNan);
    lhsImag =
        b.create<SelectOp>(rhsIsInfAndLhsImagIsNan,
                           b.create<math::CopySignOp>(zero, lhsImag), lhsImag);
    Value recalc = b.create<arith::OrIOp>(lhsIsInf, rhsIsInf);

    // Case 3. One of the pairwise products of left hand side with right hand
    // side is infinite.
    Value lhsRealTimesRhsRealIsInf = b.create<arith::CmpFOp>(
        arith::CmpFPredicate::OEQ, lhsRealTimesRhsRealAbs, inf);
    Value lhsImagTimesRhsImagIsInf = b.create<arith::CmpFOp>(
        arith::CmpFPredicate::OEQ, lhsImagTimesRhsImagAbs, inf);
    Value isSpecialCase = b.create<arith::OrIOp>(lhsRealTimesRhsRealIsInf,
                                                 lhsImagTimesRhsImagIsInf);
    Value lhsRealTimesRhsImagIsInf = b.create<arith::CmpFOp>(
        arith::CmpFPredicate::OEQ, lhsRealTimesRhsImagAbs, inf);
    isSpecialCase =
        b.create<arith::OrIOp>(isSpecialCase, lhsRealTimesRhsImagIsInf);
    Value lhsImagTimesRhsRealIsInf = b.create<arith::CmpFOp>(
        arith::CmpFPredicate::OEQ, lhsImagTimesRhsRealAbs, inf);
    isSpecialCase =
        b.create<arith::OrIOp>(isSpecialCase, lhsImagTimesRhsRealIsInf);
    Type i1Type = b.getI1Type();
    Value notRecalc = b.create<arith::XOrIOp>(
        recalc,
        b.create<arith::ConstantOp>(i1Type, b.getIntegerAttr(i1Type, 1)));
    isSpecialCase = b.create<arith::AndIOp>(isSpecialCase, notRecalc);
    Value isSpecialCaseAndLhsRealIsNan =
        b.create<arith::AndIOp>(isSpecialCase, lhsRealIsNan);
    lhsReal =
        b.create<SelectOp>(isSpecialCaseAndLhsRealIsNan,
                           b.create<math::CopySignOp>(zero, lhsReal), lhsReal);
    Value isSpecialCaseAndLhsImagIsNan =
        b.create<arith::AndIOp>(isSpecialCase, lhsImagIsNan);
    lhsImag =
        b.create<SelectOp>(isSpecialCaseAndLhsImagIsNan,
                           b.create<math::CopySignOp>(zero, lhsImag), lhsImag);
    Value isSpecialCaseAndRhsRealIsNan =
        b.create<arith::AndIOp>(isSpecialCase, rhsRealIsNan);
    rhsReal =
        b.create<SelectOp>(isSpecialCaseAndRhsRealIsNan,
                           b.create<math::CopySignOp>(zero, rhsReal), rhsReal);
    Value isSpecialCaseAndRhsImagIsNan =
        b.create<arith::AndIOp>(isSpecialCase, rhsImagIsNan);
    rhsImag =
        b.create<SelectOp>(isSpecialCaseAndRhsImagIsNan,
                           b.create<math::CopySignOp>(zero, rhsImag), rhsImag);
    recalc = b.create<arith::OrIOp>(recalc, isSpecialCase);
    recalc = b.create<arith::AndIOp>(isNan, recalc);

    // Recalculate real part.
    lhsRealTimesRhsReal = b.create<arith::MulFOp>(lhsReal, rhsReal);
    lhsImagTimesRhsImag = b.create<arith::MulFOp>(lhsImag, rhsImag);
    Value newReal =
        b.create<arith::SubFOp>(lhsRealTimesRhsReal, lhsImagTimesRhsImag);
    real =
        b.create<SelectOp>(recalc, b.create<arith::MulFOp>(inf, newReal), real);

    // Recalculate imag part.
    lhsImagTimesRhsReal = b.create<arith::MulFOp>(lhsImag, rhsReal);
    lhsRealTimesRhsImag = b.create<arith::MulFOp>(lhsReal, rhsImag);
    Value newImag =
        b.create<arith::AddFOp>(lhsImagTimesRhsReal, lhsRealTimesRhsImag);
    imag =
        b.create<SelectOp>(recalc, b.create<arith::MulFOp>(inf, newImag), imag);

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, real, imag);
    return success();
  }
};

struct NegOpConversion : public OpConversionPattern<complex::NegOp> {
  using OpConversionPattern<complex::NegOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = adaptor.getComplex().getType().cast<ComplexType>();
    auto elementType = type.getElementType().cast<FloatType>();

    Value real =
        rewriter.create<complex::ReOp>(loc, elementType, adaptor.getComplex());
    Value imag =
        rewriter.create<complex::ImOp>(loc, elementType, adaptor.getComplex());
    Value negReal = rewriter.create<arith::NegFOp>(loc, real);
    Value negImag = rewriter.create<arith::NegFOp>(loc, imag);
    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, negReal, negImag);
    return success();
  }
};

struct SignOpConversion : public OpConversionPattern<complex::SignOp> {
  using OpConversionPattern<complex::SignOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::SignOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = adaptor.getComplex().getType().cast<ComplexType>();
    auto elementType = type.getElementType().cast<FloatType>();
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value real = b.create<complex::ReOp>(elementType, adaptor.getComplex());
    Value imag = b.create<complex::ImOp>(elementType, adaptor.getComplex());
    Value zero =
        b.create<arith::ConstantOp>(elementType, b.getZeroAttr(elementType));
    Value realIsZero =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, real, zero);
    Value imagIsZero =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, imag, zero);
    Value isZero = b.create<arith::AndIOp>(realIsZero, imagIsZero);
    auto abs = b.create<complex::AbsOp>(elementType, adaptor.getComplex());
    Value realSign = b.create<arith::DivFOp>(real, abs);
    Value imagSign = b.create<arith::DivFOp>(imag, abs);
    Value sign = b.create<complex::CreateOp>(type, realSign, imagSign);
    rewriter.replaceOpWithNewOp<SelectOp>(op, isZero, adaptor.getComplex(),
                                          sign);
    return success();
  }
};
} // namespace

void mlir::populateComplexToStandardConversionPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      AbsOpConversion,
      ComparisonOpConversion<complex::EqualOp, arith::CmpFPredicate::OEQ>,
      ComparisonOpConversion<complex::NotEqualOp, arith::CmpFPredicate::UNE>,
      BinaryComplexOpConversion<complex::AddOp, arith::AddFOp>,
      BinaryComplexOpConversion<complex::SubOp, arith::SubFOp>,
      DivOpConversion,
      ExpOpConversion,
      LogOpConversion,
      Log1pOpConversion,
      MulOpConversion,
      NegOpConversion,
      SignOpConversion>(patterns.getContext());
  // clang-format on
}

namespace {
struct ConvertComplexToStandardPass
    : public ConvertComplexToStandardBase<ConvertComplexToStandardPass> {
  void runOnOperation() override;
};

void ConvertComplexToStandardPass::runOnOperation() {
  auto function = getOperation();

  // Convert to the Standard dialect using the converter defined above.
  RewritePatternSet patterns(&getContext());
  populateComplexToStandardConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithmeticDialect, StandardOpsDialect,
                         math::MathDialect>();
  target.addLegalOp<complex::CreateOp, complex::ImOp, complex::ReOp>();
  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
}
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createConvertComplexToStandardPass() {
  return std::make_unique<ConvertComplexToStandardPass>();
}
