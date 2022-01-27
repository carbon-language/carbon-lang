//===- TosaToLinalgNamed.cpp - Lowering Tosa to Linalg Named Ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the Linalg named ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

using namespace mlir;

static SmallVector<StringRef> getNParallelLoopsAttrs(unsigned nParallelLoops) {
  return SmallVector<StringRef>(nParallelLoops, getParallelIteratorTypeName());
}

template <typename T>
static void getValuesFromIntArrayAttribute(ArrayAttr attr,
                                           SmallVector<T> &arrayValues) {
  for (Attribute val : attr.getValue()) {
    arrayValues.push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
  }
}

template <typename T, typename P>
static mlir::SelectOp clampHelper(Location loc, Value arg,
                                  arith::ConstantOp min, arith::ConstantOp max,
                                  P pred, OpBuilder &rewriter) {
  auto smallerThanMin = rewriter.create<T>(loc, pred, arg, min);
  auto minOrArg =
      rewriter.create<mlir::SelectOp>(loc, smallerThanMin, min, arg);
  auto largerThanMax = rewriter.create<T>(loc, pred, max, arg);
  return rewriter.create<mlir::SelectOp>(loc, largerThanMax, max, minOrArg);
}

static mlir::Value applyPad(Location loc, Value input, ArrayRef<int64_t> pad,
                            Attribute padAttr, OpBuilder &rewriter) {
  // Input should be padded if necessary.
  if (llvm::all_of(pad, [](int64_t p) { return p == 0; }))
    return input;

  ShapedType inputTy = input.getType().cast<ShapedType>();
  Type inputETy = inputTy.getElementType();
  auto inputShape = inputTy.getShape();

  assert((inputShape.size() * 2) == pad.size());

  SmallVector<int64_t, 4> paddedShape;
  SmallVector<OpFoldResult, 8> lowIndices;
  SmallVector<OpFoldResult, 8> highIndices;
  for (int i = 0, s = inputShape.size(); i < s; i++) {
    auto lowPad = pad[i * 2];
    auto highPad = pad[i * 2 + 1];
    paddedShape.push_back(inputShape[i] + highPad + lowPad);
    lowIndices.push_back(rewriter.getIndexAttr(lowPad));
    highIndices.push_back(rewriter.getIndexAttr(highPad));
  }

  Value padValue = rewriter.create<arith::ConstantOp>(loc, padAttr);

  return linalg::PadTensorOp::createPadScalarOp(
             RankedTensorType::get(paddedShape, inputETy), input, padValue,
             lowIndices, highIndices, /*nofold=*/false, loc, rewriter)
      .result();
}

static SmallVector<Value> filterDynamicDims(const SmallVector<Value> &dynDims) {
  SmallVector<Value> filteredDims;
  for (auto dim : dynDims)
    if (dim)
      filteredDims.push_back(dim);
  return filteredDims;
}

namespace {

class ConvConverter : public OpConversionPattern<tosa::Conv2DOp> {
public:
  using OpConversionPattern<tosa::Conv2DOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = op->getOperand(0);
    Value weight = op->getOperand(1);
    Value bias = op->getOperand(2);

    ShapedType inputTy = input.getType().cast<ShapedType>();
    ShapedType weightTy = weight.getType().cast<ShapedType>();
    ShapedType biasTy = bias.getType().cast<ShapedType>();
    ShapedType resultTy = op->getResult(0).getType().cast<ShapedType>();

    Type inputETy = inputTy.getElementType();
    Type resultETy = resultTy.getElementType();

    auto padAttr = op->getAttr("pad").cast<ArrayAttr>();
    auto strideTosaAttr = op->getAttr("stride").cast<ArrayAttr>();
    auto dilationTosaAttr = op->getAttr("dilation").cast<ArrayAttr>();
    bool isQuantized = op->hasAttr("quantization_info");

    if (!inputTy.hasStaticShape() || !weightTy.hasStaticShape() ||
        !biasTy.hasStaticShape() || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op,
                                         "tosa.conv ops require static shapes");

    if (inputETy.isUnsignedInteger())
      return rewriter.notifyMatchFailure(
          op, "tosa.conv ops does not support unsigned integer input");

    auto weightShape = weightTy.getShape();

    // Apply padding as necessary.
    Attribute zeroAttr = rewriter.getZeroAttr(inputETy);
    if (isQuantized) {
      auto quantizationInfo =
          op->getAttr("quantization_info").cast<tosa::ConvOpQuantizationAttr>();
      auto iZp = quantizationInfo.input_zp().getValue().getSExtValue();

      int64_t intMin =
          APInt::getSignedMinValue(inputETy.getIntOrFloatBitWidth())
              .getSExtValue();
      int64_t intMax =
          APInt::getSignedMaxValue(inputETy.getIntOrFloatBitWidth())
              .getSExtValue();

      if (iZp < intMin || iZp > intMax)
        return rewriter.notifyMatchFailure(
            op, "tosa.conv op quantization has zp outside of input range");

      zeroAttr = rewriter.getIntegerAttr(inputETy, iZp);
    }

    llvm::SmallVector<int64_t> pad;
    pad.resize(2, 0);
    getValuesFromIntArrayAttribute(padAttr, pad);
    pad.resize(pad.size() + 2, 0);
    input = applyPad(loc, input, pad, zeroAttr, rewriter);

    // Transpose the kernel to match dimension ordering of the linalg
    // convolution operation.
    // TODO(suderman): See if this can be efficiently folded - check whether
    // the input is used anywhere else, if not fold the constant.
    SmallVector<int64_t> weightPerm{1, 2, 3, 0};
    SmallVector<int64_t> newWeightShape{weightShape[1], weightShape[2],
                                        weightShape[3], weightShape[0]};
    auto weightPermAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({4}, rewriter.getI64Type()), weightPerm);
    Value weightPermValue =
        rewriter.create<arith::ConstantOp>(loc, weightPermAttr);
    Type newWeightTy =
        RankedTensorType::get(newWeightShape, weightTy.getElementType());
    weight = rewriter.create<tosa::TransposeOp>(loc, newWeightTy, weight,
                                                weightPermValue);

    Attribute resultZeroAttr = rewriter.getZeroAttr(resultETy);
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultTy.getShape(), resultETy);
    Value zero = rewriter.create<arith::ConstantOp>(loc, resultZeroAttr);
    Value zeroTensor =
        rewriter.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);

    // Extract the attributes for convolution.
    llvm::SmallVector<int64_t> stride, dilation;
    getValuesFromIntArrayAttribute(strideTosaAttr, stride);
    getValuesFromIntArrayAttribute(dilationTosaAttr, dilation);

    // Create the convolution op.
    auto strideAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), stride);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilation);

    // Create maps for the bias broadcasting
    SmallVector<AffineMap, 4> indexingMaps;
    indexingMaps.push_back(AffineMap::get(
        /*dimCount=*/resultTy.getRank(), /*symbolCount=*/0,
        {rewriter.getAffineDimExpr(3)}, rewriter.getContext()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultTy.getRank()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultTy.getRank()));

    Value biasInitTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultTy.getShape(), resultETy);

    if (isQuantized) {
      auto quantizationInfo =
          op->getAttr("quantization_info").cast<tosa::ConvOpQuantizationAttr>();
      auto iZp = rewriter.getI32IntegerAttr(
          quantizationInfo.input_zp().getValue().getSExtValue());
      auto kZp = rewriter.getI32IntegerAttr(
          quantizationInfo.weight_zp().getValue().getSExtValue());

      auto iZpVal = rewriter.create<arith::ConstantOp>(loc, iZp);
      auto kZpVal = rewriter.create<arith::ConstantOp>(loc, kZp);
      Value conv =
          rewriter
              .create<linalg::Conv2DNhwcHwcfQOp>(
                  loc, resultTy, ValueRange{input, weight, iZpVal, kZpVal},
                  ValueRange{zeroTensor}, strideAttr, dilationAttr)
              ->getResult(0);

      Value result =
          rewriter
              .create<linalg::GenericOp>(
                  loc, resultTy, ValueRange({bias, conv}), biasInitTensor,
                  indexingMaps, getNParallelLoopsAttrs(resultTy.getRank()),
                  [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      ValueRange args) {
                    Value added = nestedBuilder.create<arith::AddIOp>(
                        loc, args[0], args[1]);
                    nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
                  })
              .getResult(0);
      rewriter.replaceOp(op, result);
      return success();
    }

    Value conv = rewriter
                     .create<linalg::Conv2DNhwcHwcfOp>(
                         loc, resultTy, ValueRange{input, weight},
                         ValueRange{zeroTensor}, strideAttr, dilationAttr)
                     ->getResult(0);

    Value result =
        rewriter
            .create<linalg::GenericOp>(
                loc, resultTy, ValueRange({bias, conv}), biasInitTensor,
                indexingMaps, getNParallelLoopsAttrs(resultTy.getRank()),
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange args) {
                  Value added = nestedBuilder.create<arith::AddFOp>(
                      loc, args[0], args[1]);
                  nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
                })
            .getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

class DepthwiseConvConverter
    : public OpConversionPattern<tosa::DepthwiseConv2DOp> {
public:
  using OpConversionPattern<tosa::DepthwiseConv2DOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::DepthwiseConv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = op->getOperand(0);
    Value weight = op->getOperand(1);
    Value bias = op->getOperand(2);

    ShapedType inputTy = input.getType().cast<ShapedType>();
    ShapedType weightTy = weight.getType().cast<ShapedType>();
    ShapedType biasTy = bias.getType().cast<ShapedType>();
    ShapedType resultTy = op->getResult(0).getType().cast<ShapedType>();

    Type inputETy = inputTy.getElementType();
    Type resultETy = resultTy.getElementType();

    auto padAttr = op->getAttr("pad").cast<ArrayAttr>();
    auto strideTosaAttr = op->getAttr("stride").cast<ArrayAttr>();
    auto dilationTosaAttr = op->getAttr("dilation").cast<ArrayAttr>();

    bool isQuantized = op->hasAttr("quantization_info");
    IntegerAttr iZp;
    IntegerAttr kZp;
    if (isQuantized) {
      auto quantizationInfo =
          op->getAttr("quantization_info").cast<tosa::ConvOpQuantizationAttr>();
      iZp = rewriter.getI32IntegerAttr(
          quantizationInfo.input_zp().getValue().getSExtValue());
      kZp = rewriter.getI32IntegerAttr(
          quantizationInfo.weight_zp().getValue().getSExtValue());
    }

    if (!inputTy.hasStaticShape() || !weightTy.hasStaticShape() ||
        !biasTy.hasStaticShape() || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op,
                                         "tosa.conv ops require static shapes");

    auto weightShape = weightTy.getShape();
    auto resultShape = resultTy.getShape();

    // Apply padding as necessary.
    Attribute zeroAttr = rewriter.getZeroAttr(inputETy);
    if (isQuantized) {
      auto quantizationInfo =
          op->getAttr("quantization_info").cast<tosa::ConvOpQuantizationAttr>();
      auto iZp = quantizationInfo.input_zp().getValue().getSExtValue();

      int64_t intMin =
          APInt::getSignedMinValue(inputETy.getIntOrFloatBitWidth())
              .getSExtValue();
      int64_t intMax =
          APInt::getSignedMaxValue(inputETy.getIntOrFloatBitWidth())
              .getSExtValue();

      if (iZp < intMin || iZp > intMax)
        return rewriter.notifyMatchFailure(
            op, "tosa.depthwise_conv op quantization has zp outside of input "
                "range");

      zeroAttr = rewriter.getIntegerAttr(inputETy, iZp);
    }

    llvm::SmallVector<int64_t> pad;
    pad.resize(2, 0);
    getValuesFromIntArrayAttribute(padAttr, pad);
    pad.resize(pad.size() + 2, 0);

    input = applyPad(loc, input, pad, zeroAttr, rewriter);

    // Extract the attributes for convolution.
    llvm::SmallVector<int64_t> stride, dilation;
    getValuesFromIntArrayAttribute(strideTosaAttr, stride);
    getValuesFromIntArrayAttribute(dilationTosaAttr, dilation);

    // Create the convolution op.
    auto strideAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), stride);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilation);
    ShapedType linalgConvTy =
        RankedTensorType::get({resultShape[0], resultShape[1], resultShape[2],
                               weightShape[2], weightShape[3]},
                              resultETy);

    // Broadcast the initial value to the output tensor before convolving.
    SmallVector<AffineMap, 4> indexingMaps;
    indexingMaps.push_back(AffineMap::get(
        /*dimCount=*/resultTy.getRank(), /*symbolCount=*/0,
        {rewriter.getAffineDimExpr(3)}, rewriter.getContext()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultTy.getRank()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultTy.getRank()));

    Attribute resultZeroAttr = rewriter.getZeroAttr(resultETy);
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, linalgConvTy.getShape(), resultETy);
    Value zero = rewriter.create<arith::ConstantOp>(loc, resultZeroAttr);
    Value zeroTensor =
        rewriter.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);

    Value biasInitTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultTy.getShape(), resultETy);
    if (!isQuantized) {
      Value conv = rewriter
                       .create<linalg::DepthwiseConv2DNhwcHwcmOp>(
                           loc, linalgConvTy, ValueRange{input, weight},
                           ValueRange{zeroTensor}, strideAttr, dilationAttr)
                       .getResult(0);
      Value convReshape = rewriter.create<tosa::ReshapeOp>(
          loc, resultTy, conv, rewriter.getI64ArrayAttr(resultTy.getShape()));
      Value result =
          rewriter
              .create<linalg::GenericOp>(
                  loc, resultTy, ValueRange({bias, convReshape}),
                  biasInitTensor, indexingMaps,
                  getNParallelLoopsAttrs(resultTy.getRank()),
                  [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      ValueRange args) {
                    Value added = nestedBuilder.create<arith::AddFOp>(
                        loc, args[0], args[1]);
                    nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
                  })
              .getResult(0);
      rewriter.replaceOp(op, result);
    } else {
      auto iZpVal = rewriter.create<arith::ConstantOp>(loc, iZp);
      auto kZpVal = rewriter.create<arith::ConstantOp>(loc, kZp);
      Value conv =
          rewriter
              .create<linalg::DepthwiseConv2DNhwcHwcmQOp>(
                  loc, linalgConvTy, ValueRange{input, weight, iZpVal, kZpVal},
                  ValueRange{zeroTensor}, strideAttr, dilationAttr)
              .getResult(0);
      Value convReshape = rewriter.create<tosa::ReshapeOp>(
          loc, resultTy, conv, rewriter.getI64ArrayAttr(resultTy.getShape()));
      Value result =
          rewriter
              .create<linalg::GenericOp>(
                  loc, resultTy, ValueRange({bias, convReshape}),
                  biasInitTensor, indexingMaps,
                  getNParallelLoopsAttrs(resultTy.getRank()),
                  [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      ValueRange args) {
                    Value added = nestedBuilder.create<arith::AddIOp>(
                        loc, args[0], args[1]);
                    nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
                  })
              .getResult(0);
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

class MatMulConverter : public OpConversionPattern<tosa::MatMulOp> {
public:
  using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    auto outputTy = op.getType().cast<ShapedType>();
    auto outputElementTy = outputTy.getElementType();

    auto firstOperandTy = op->getOperand(0).getType().cast<ShapedType>();
    auto secondOperandTy = op->getOperand(1).getType().cast<ShapedType>();

    SmallVector<Value> dynDims;
    dynDims.resize(op->getResult(0).getType().cast<ShapedType>().getRank());

    if (!firstOperandTy.hasRank() || firstOperandTy.isDynamicDim(0)) {
      dynDims[0] = rewriter.create<tensor::DimOp>(loc, op->getOperand(0), 0);
    }

    if (!firstOperandTy.hasRank() || firstOperandTy.isDynamicDim(1)) {
      dynDims[1] = rewriter.create<tensor::DimOp>(loc, op->getOperand(0), 1);
    }

    if (!secondOperandTy.hasRank() || secondOperandTy.isDynamicDim(2)) {
      dynDims[2] = rewriter.create<tensor::DimOp>(loc, op->getOperand(1), 2);
    }

    SmallVector<Value> filteredDims = filterDynamicDims(dynDims);

    auto zeroAttr = rewriter.getZeroAttr(outputElementTy);
    Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, filteredDims, outputTy.getShape(), outputTy.getElementType());
    Value zeroTensor =
        rewriter.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);
    if (!op.quantization_info()) {
      rewriter.replaceOpWithNewOp<linalg::BatchMatmulOp>(
          op, TypeRange{op.getType()}, ValueRange{adaptor.a(), adaptor.b()},
          ValueRange{zeroTensor});
      return success();
    }

    auto quantizationInfo = op.quantization_info().getValue();
    auto aZp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(
                 quantizationInfo.a_zp().getValue().getSExtValue()));
    auto bZp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(
                 quantizationInfo.b_zp().getValue().getSExtValue()));
    rewriter.replaceOpWithNewOp<linalg::QuantizedBatchMatmulOp>(
        op, TypeRange{op.getType()},
        ValueRange{adaptor.a(), adaptor.b(), aZp, bZp}, zeroTensor);

    return success();
  }
};

class FullyConnectedConverter
    : public OpConversionPattern<tosa::FullyConnectedOp> {
public:
  using OpConversionPattern<tosa::FullyConnectedOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::FullyConnectedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto outputTy = op.getType().cast<ShapedType>();
    auto input = op.input();
    auto inputTy = input.getType().cast<ShapedType>();

    auto bias = op.bias();

    auto weight = op.weight();
    auto weightTy = weight.getType().cast<ShapedType>();
    auto weightShape = weightTy.getShape();

    auto outputETy = outputTy.getElementType();

    SmallVector<Value> dynDims;
    dynDims.resize(op->getResult(0).getType().cast<ShapedType>().getRank());

    if (!inputTy.hasRank() || inputTy.isDynamicDim(0)) {
      dynDims[0] = rewriter.create<tensor::DimOp>(loc, input, 0);
    }

    if (!weightTy.hasRank() || weightTy.isDynamicDim(0)) {
      dynDims[1] = rewriter.create<tensor::DimOp>(loc, weight, 0);
    }

    SmallVector<Value> filteredDims = filterDynamicDims(dynDims);

    // Creating maps for the output of MatMul and the bias
    SmallVector<AffineMap, 4> indexingMaps;

    // Broadcast the bias.
    indexingMaps.push_back(AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                          {rewriter.getAffineDimExpr(1)},
                                          rewriter.getContext()));

    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(outputTy.getRank()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(outputTy.getRank()));

    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, filteredDims, outputTy.getShape(), outputTy.getElementType());

    // When quantized, the input elemeny type is not the same as the output
    Attribute resultZeroAttr = rewriter.getZeroAttr(outputETy);
    Value zero = rewriter.create<arith::ConstantOp>(loc, resultZeroAttr);
    Value zeroTensor =
        rewriter.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);

    SmallVector<int64_t> permutation{1, 0};
    auto permutationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), permutation);
    Value permutationValue =
        rewriter.create<arith::ConstantOp>(loc, permutationAttr);

    SmallVector<int64_t> newWeightShape{weightShape[1], weightShape[0]};
    Type newWeightTy =
        RankedTensorType::get(newWeightShape, weightTy.getElementType());

    Value transposedWeight = rewriter.create<tosa::TransposeOp>(
        loc, newWeightTy, weight, permutationValue);

    auto biasInitTensor =
        rewriter
            .create<linalg::InitTensorOp>(loc, filteredDims,
                                          outputTy.getShape(), outputETy)
            ->getResults();

    if (!op.quantization_info()) {
      Value matmul = rewriter
                         .create<linalg::MatmulOp>(
                             loc, TypeRange{op.getType()},
                             ValueRange{input, transposedWeight}, zeroTensor)
                         ->getResult(0);

      Value result =
          rewriter
              .create<linalg::GenericOp>(
                  loc, outputTy, ValueRange({bias, matmul}), biasInitTensor,
                  indexingMaps, getNParallelLoopsAttrs(outputTy.getRank()),
                  [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      ValueRange args) {
                    Value added = nestedBuilder.create<arith::AddFOp>(
                        loc, args[0], args[1]);
                    nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
                  })
              .getResult(0);
      rewriter.replaceOp(op, result);
      return success();
    }

    auto quantizationInfo = op.quantization_info().getValue();
    auto inputZp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(
                 quantizationInfo.input_zp().getValue().getSExtValue()));
    auto outputZp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(
                 quantizationInfo.weight_zp().getValue().getSExtValue()));
    Value matmul =
        rewriter
            .create<linalg::QuantizedMatmulOp>(
                loc, TypeRange{op.getType()},
                ValueRange{input, transposedWeight, inputZp, outputZp},
                zeroTensor)
            ->getResult(0);
    Value result =
        rewriter
            .create<linalg::GenericOp>(
                loc, outputTy, ValueRange({bias, matmul}), biasInitTensor,
                indexingMaps, getNParallelLoopsAttrs(outputTy.getRank()),
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange args) {
                  Value added = nestedBuilder.create<arith::AddIOp>(
                      loc, args[0], args[1]);
                  nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
                })
            .getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class MaxPool2dConverter : public OpRewritePattern<tosa::MaxPool2dOp> {
public:
  using OpRewritePattern<tosa::MaxPool2dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MaxPool2dOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value input = op.input();
    ShapedType inputTy = input.getType().cast<ShapedType>();

    ShapedType resultTy = op.getType().template cast<ShapedType>();
    Type resultETy = inputTy.getElementType();

    if (!inputTy.hasStaticShape())
      return failure();

    // Determine what the initial value needs to be for the max pool op.
    Attribute initialAttr;
    if (resultETy.isF32())
      initialAttr = rewriter.getFloatAttr(
          resultETy,
          APFloat::getLargest(resultETy.cast<FloatType>().getFloatSemantics(),
                              true));

    if (resultETy.isa<IntegerType>())
      initialAttr = rewriter.getIntegerAttr(
          resultETy,
          APInt::getSignedMinValue(resultETy.getIntOrFloatBitWidth()));

    if (!initialAttr)
      return rewriter.notifyMatchFailure(
          op, "Unsupported initial value for tosa.maxpool_2d op");

    // Apply padding as necessary.
    llvm::SmallVector<int64_t> pad;
    pad.resize(2, 0);
    getValuesFromIntArrayAttribute(op.pad(), pad);
    pad.resize(pad.size() + 2, 0);
    Value paddedInput = applyPad(loc, input, pad, initialAttr, rewriter);

    Value initialValue = rewriter.create<arith::ConstantOp>(loc, initialAttr);

    SmallVector<int64_t> kernel, stride;
    getValuesFromIntArrayAttribute(op.kernel(), kernel);
    getValuesFromIntArrayAttribute(op.stride(), stride);

    Attribute strideAttr = rewriter.getI64VectorAttr(stride);
    Attribute dilationAttr = rewriter.getI64VectorAttr({1, 1});

    // Create the linalg op that performs pooling.
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultTy.getShape(), resultTy.getElementType());

    Value filledInitTensor =
        rewriter.create<linalg::FillOp>(loc, initialValue, initTensor).result();

    Value fakeWindowDims =
        rewriter.create<linalg::InitTensorOp>(loc, kernel, resultETy);

    rewriter.replaceOpWithNewOp<linalg::PoolingNhwcMaxOp>(
        op, ArrayRef<Type>{resultTy}, ValueRange{paddedInput, fakeWindowDims},
        filledInitTensor, strideAttr, dilationAttr);
    return success();
  }
};

class AvgPool2dConverter : public OpRewritePattern<tosa::AvgPool2dOp> {
public:
  using OpRewritePattern<tosa::AvgPool2dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::AvgPool2dOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value input = op.input();
    ShapedType inputTy = input.getType().cast<ShapedType>();
    Type inElementTy = inputTy.getElementType();

    ShapedType resultTy = op.getType().template cast<ShapedType>();
    Type resultETy = op.getType().cast<ShapedType>().getElementType();

    Type accETy =
        inElementTy.isa<IntegerType>() ? rewriter.getI32Type() : inElementTy;
    ShapedType accTy = resultTy.clone(accETy);

    if (!inputTy.hasStaticShape())
      return failure();

    // Apply padding as necessary.
    llvm::SmallVector<int64_t> pad;
    pad.resize(2, 0);
    getValuesFromIntArrayAttribute(op.pad(), pad);
    pad.resize(pad.size() + 2, 0);
    Attribute padAttr = rewriter.getZeroAttr(inElementTy);
    Value paddedInput = applyPad(loc, input, pad, padAttr, rewriter);

    Attribute initialAttr = rewriter.getZeroAttr(accETy);
    Value initialValue = rewriter.create<arith::ConstantOp>(loc, initialAttr);

    SmallVector<int64_t> kernel, stride;
    getValuesFromIntArrayAttribute(op.kernel(), kernel);
    getValuesFromIntArrayAttribute(op.stride(), stride);

    Attribute strideAttr = rewriter.getI64VectorAttr(stride);
    Attribute dilationAttr = rewriter.getI64VectorAttr({1, 1});

    // Create the linalg op that performs pooling.
    Value poolInitTensor =
        rewriter.create<linalg::InitTensorOp>(loc, accTy.getShape(), accETy);

    Value filledInitTensor =
        rewriter.create<linalg::FillOp>(loc, initialValue, poolInitTensor)
            .result();

    Value fakeWindowDims =
        rewriter.create<linalg::InitTensorOp>(loc, kernel, accETy);

    // Sum across the pooled region.
    Value poolingOp = rewriter
                          .create<linalg::PoolingNhwcSumOp>(
                              loc, ArrayRef<Type>{accTy},
                              ValueRange{paddedInput, fakeWindowDims},
                              filledInitTensor, strideAttr, dilationAttr)
                          .getResult(0);

    // Normalize the summed value by the number of elements grouped in each
    // pool.
    auto poolingOpTy = poolingOp.getType().cast<ShapedType>();
    auto affineMap = rewriter.getMultiDimIdentityMap(resultTy.getRank());

    Value genericInitTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultTy.getShape(), resultETy);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, ArrayRef<Type>({resultTy}), ValueRange{poolingOp},
        ValueRange{genericInitTensor},
        ArrayRef<AffineMap>({affineMap, affineMap}),
        getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
          auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
          auto iH = rewriter.create<arith::ConstantIndexOp>(
              loc, poolingOpTy.getDimSize(1) - 1);
          auto iW = rewriter.create<arith::ConstantIndexOp>(
              loc, poolingOpTy.getDimSize(2) - 1);

          // Compute the indices from either end.
          auto y0 = rewriter.create<linalg::IndexOp>(loc, 1);
          auto x0 = rewriter.create<linalg::IndexOp>(loc, 2);
          auto y1 = rewriter.create<arith::SubIOp>(loc, iH, y0);
          auto x1 = rewriter.create<arith::SubIOp>(loc, iW, x0);

          // Determines what the portion of valid input is covered by the
          // kernel.
          auto padFn = [&](Value v, Value x, int64_t pad) -> Value {
            if (pad == 0)
              return v;

            auto padVal = rewriter.create<arith::ConstantIndexOp>(loc, pad);
            Value dx = rewriter.create<arith::SubIOp>(loc, x, padVal);

            Value cmp = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, dx, zero);
            Value offset = rewriter.create<mlir::SelectOp>(loc, cmp, dx, zero);
            return rewriter.create<arith::AddIOp>(loc, v, offset)->getResult(0);
          };

          // Compute the vertical component of coverage.
          auto kH0 = rewriter.create<arith::ConstantIndexOp>(loc, kernel[0]);
          auto kH1 = padFn(kH0, y0, pad[2]);
          auto kH2 = padFn(kH1, y1, pad[3]);
          auto kHCmp = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, kH2, one);
          auto kH3 = rewriter.create<SelectOp>(loc, kHCmp, one, kH2);

          // compute the horizontal component of coverage.
          auto kW0 = rewriter.create<arith::ConstantIndexOp>(loc, kernel[1]);
          auto kW1 = padFn(kW0, x0, pad[4]);
          auto kW2 = padFn(kW1, x1, pad[5]);
          auto kWCmp = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, kW2, one);
          auto kW3 = rewriter.create<SelectOp>(loc, kWCmp, one, kW2);

          // Compute the total number of elements and normalize.
          Value count = rewriter.create<arith::MulIOp>(loc, kH3, kW3);
          auto countI = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getI32Type(), count);

          // Divide by the number of summed values. For floats this is just
          // a div however for quantized values input normalization had
          // to be applied.
          Value poolVal = args[0];
          if (accETy.isa<FloatType>()) {
            auto countF = rewriter.create<arith::SIToFPOp>(loc, accETy, countI);
            poolVal = rewriter.create<arith::DivFOp>(loc, poolVal, countF)
                          ->getResult(0);
          } else {

            // If we have quantization information we need to apply an offset
            // for the input zp value.
            if (op.quantization_info()) {
              auto quantizationInfo = op.quantization_info().getValue();
              auto inputZp = rewriter.create<arith::ConstantOp>(
                  loc, quantizationInfo.input_zp());
              Value offset =
                  rewriter.create<arith::MulIOp>(loc, accETy, countI, inputZp);
              poolVal =
                  rewriter.create<arith::SubIOp>(loc, accETy, poolVal, offset);
            }

            // Compute the multiplier and shift values for the quantization
            // normalization. Preferably we would want to compute more bits
            // however 32-bits should be enough for compute. Honestly we
            // should probably straight divide.
            int64_t numerator = ((1 << 30) + 1);
            int64_t shift = 30;

            Value numeratorVal = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI32IntegerAttr(numerator));
            Value multiplierVal =
                rewriter
                    .create<arith::DivUIOp>(loc, rewriter.getI32Type(),
                                            numeratorVal, countI)
                    .getResult();
            Value shiftVal = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI8IntegerAttr(shift));

            auto scaled =
                rewriter
                    .create<tosa::ApplyScaleOp>(
                        loc, rewriter.getI32Type(), poolVal, multiplierVal,
                        shiftVal, rewriter.getBoolAttr(false))
                    .getResult();

            // If we have quantization information we need to apply output
            // zeropoint.
            if (op.quantization_info()) {
              auto quantizationInfo = op.quantization_info().getValue();
              auto outputZp = rewriter.create<arith::ConstantOp>(
                  loc, quantizationInfo.output_zp());
              scaled = rewriter.create<arith::AddIOp>(loc, scaled, outputZp)
                           .getResult();
            }

            // Apply Clip.
            int64_t outBitwidth = resultETy.getIntOrFloatBitWidth();

            auto min = rewriter.create<arith::ConstantIntOp>(
                loc, APInt::getSignedMinValue(outBitwidth).getSExtValue(),
                accETy);
            auto max = rewriter.create<arith::ConstantIntOp>(
                loc, APInt::getSignedMaxValue(outBitwidth).getSExtValue(),
                accETy);
            auto clamp = clampHelper<arith::CmpIOp>(
                loc, scaled, min, max, arith::CmpIPredicate::slt, rewriter);

            poolVal = clamp;
            // Convert type.
            if (resultETy != clamp.getType()) {
              poolVal =
                  rewriter.create<arith::TruncIOp>(loc, resultETy, poolVal);
            }
          }

          rewriter.create<linalg::YieldOp>(loc, poolVal);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToLinalgNamedConversionPatterns(
    RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
      ConvConverter,
      DepthwiseConvConverter,
      MatMulConverter,
      MaxPool2dConverter,
      AvgPool2dConverter,
      FullyConnectedConverter>(patterns->getContext());
  // clang-format on
}
