//===- NamedOpConversions.cpp - Implements conversions between named ops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversions between named ops that can be seens as
// canonicalizations of named ops.
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::linalg;

static llvm::SmallVector<int64_t> getIndicesVector(int start, int end) {
  return llvm::to_vector<2>(llvm::seq<int64_t>(start, end));
}

static LogicalResult
matchAndReplaceDepthwiseConv(Operation *operation, Value input, Value kernel,
                             Value iZp, Value kZp, Value init, Attribute stride,
                             Attribute dilation, PatternRewriter &rewriter) {
  Location loc = operation->getLoc();
  auto linalgOp = dyn_cast<LinalgOp>(operation);
  // Exit out on the memref version of this operation.
  if (!linalgOp || !linalgOp.hasTensorSemantics())
    return failure();

  auto result = operation->getResult(0);

  auto kernelTy = kernel.getType().dyn_cast<RankedTensorType>();
  auto initTy = init.getType().dyn_cast<RankedTensorType>();
  auto resultTy = result.getType().template dyn_cast<RankedTensorType>();
  if (!kernelTy || !initTy || !resultTy)
    return failure();

  if (kernelTy.getDimSize(3) != 1)
    return failure();

  // Collapse kernel dims.
  SmallVector<ReassociationIndices, 4> collapsedKernelDims = {
      getIndicesVector(0, 1), getIndicesVector(1, 2), getIndicesVector(2, 4)};
  auto newKernelTy = RankedTensorType::get(
      {kernelTy.getDimSize(0), kernelTy.getDimSize(1), kernelTy.getDimSize(2)},
      kernelTy.getElementType());
  auto collapsedKernel = rewriter.create<tensor::CollapseShapeOp>(
      loc, newKernelTy, kernel, collapsedKernelDims);

  // Collapse init dims.
  SmallVector<ReassociationIndices, 4> collapsedInitDims = {
      getIndicesVector(0, 1), getIndicesVector(1, 2), getIndicesVector(2, 3),
      getIndicesVector(3, 5)};
  auto newInitTy =
      RankedTensorType::get({initTy.getDimSize(0), initTy.getDimSize(1),
                             initTy.getDimSize(2), initTy.getDimSize(3)},
                            initTy.getElementType());
  auto collapsedInit = rewriter.create<tensor::CollapseShapeOp>(
      loc, newInitTy, init, collapsedInitDims);

  Value newConv;
  if (isa<DepthwiseConv2DNhwcHwcmOp>(operation)) {
    newConv = rewriter
                  .create<DepthwiseConv2DNhwcHwcOp>(
                      loc, newInitTy, ValueRange{input, collapsedKernel},
                      ValueRange{collapsedInit}, stride, dilation)
                  .getResult(0);
  } else if (isa<DepthwiseConv2DNhwcHwcmQOp>(operation)) {
    newConv =
        rewriter
            .create<DepthwiseConv2DNhwcHwcQOp>(
                loc, newInitTy, ValueRange{input, collapsedKernel, iZp, kZp},
                ValueRange{collapsedInit}, stride, dilation)
            .getResult(0);
  }

  if (!newConv)
    return failure();

  // Expand dimensions back out to
  rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
      operation, resultTy, newConv, collapsedInitDims);
  return success();
}

namespace {
struct SimplifyDepthwiseConvOp
    : public OpRewritePattern<DepthwiseConv2DNhwcHwcmOp> {
  using OpRewritePattern<DepthwiseConv2DNhwcHwcmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DepthwiseConv2DNhwcHwcmOp op,
                                PatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    Value input = op.getInputOperand(0)->get();
    Value kernel = op.getInputOperand(1)->get();
    Value init = op.getOutputOperand(0)->get();

    auto stride = op.strides();
    auto dilation = op.dilations();

    return matchAndReplaceDepthwiseConv(operation, input, kernel, nullptr,
                                        nullptr, init, stride, dilation,
                                        rewriter);
  }
};

struct SimplifyDepthwiseConvQOp
    : public OpRewritePattern<DepthwiseConv2DNhwcHwcmQOp> {
  using OpRewritePattern<DepthwiseConv2DNhwcHwcmQOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DepthwiseConv2DNhwcHwcmQOp op,
                                PatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    Value input = op.getInputOperand(0)->get();
    Value kernel = op.getInputOperand(1)->get();
    Value iZp = op.getInputOperand(2)->get();
    Value kZp = op.getInputOperand(3)->get();
    Value init = op.getOutputOperand(0)->get();

    auto stride = op.strides();
    auto dilation = op.dilations();

    return matchAndReplaceDepthwiseConv(operation, input, kernel, iZp, kZp,
                                        init, stride, dilation, rewriter);
  }
};

struct LinalgNamedOpConversionPass
    : public LinalgNamedOpConversionBase<LinalgNamedOpConversionPass> {
  LinalgNamedOpConversionPass() = default;
  LinalgNamedOpConversionPass(const LinalgNamedOpConversionPass &) = default;

  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateLinalgNamedOpConversionPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void mlir::linalg::populateLinalgNamedOpConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<SimplifyDepthwiseConvOp, SimplifyDepthwiseConvQOp>(
      patterns.getContext());
}

std::unique_ptr<Pass> mlir::createLinalgNamedOpConversionPass() {
  return std::make_unique<LinalgNamedOpConversionPass>();
}
