//===- SparsificationPass.cpp - Pass for autogen spares tensor code -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

//===----------------------------------------------------------------------===//
// Passes declaration.
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Passes implementation.
//===----------------------------------------------------------------------===//

struct SparsificationPass : public SparsificationBase<SparsificationPass> {

  SparsificationPass() = default;
  SparsificationPass(const SparsificationPass &pass)
      : SparsificationBase<SparsificationPass>() {}

  /// Returns parallelization strategy given on command line.
  SparseParallelizationStrategy parallelOption() {
    switch (parallelization) {
    default:
      return SparseParallelizationStrategy::kNone;
    case 1:
      return SparseParallelizationStrategy::kDenseOuterLoop;
    case 2:
      return SparseParallelizationStrategy::kAnyStorageOuterLoop;
    case 3:
      return SparseParallelizationStrategy::kDenseAnyLoop;
    case 4:
      return SparseParallelizationStrategy::kAnyStorageAnyLoop;
    }
  }

  /// Returns vectorization strategy given on command line.
  SparseVectorizationStrategy vectorOption() {
    switch (vectorization) {
    default:
      return SparseVectorizationStrategy::kNone;
    case 1:
      return SparseVectorizationStrategy::kDenseInnerLoop;
    case 2:
      return SparseVectorizationStrategy::kAnyStorageInnerLoop;
    }
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    // Translate strategy flags to strategy options.
    SparsificationOptions options(parallelOption(), vectorOption(),
                                  vectorLength, enableSIMDIndex32);
    // Apply rewriting.
    populateSparsificationPatterns(patterns, options);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

class SparseTensorTypeConverter : public TypeConverter {
public:
  SparseTensorTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertSparseTensorTypes);
  }
  // Maps each sparse tensor type to an opaque pointer.
  static Optional<Type> convertSparseTensorTypes(Type type) {
    if (getSparseTensorEncoding(type) != nullptr)
      return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
    return llvm::None;
  }
};

struct SparseTensorConversionPass
    : public SparseTensorConversionBase<SparseTensorConversionPass> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    SparseTensorTypeConverter converter;
    ConversionTarget target(*ctx);
    target.addIllegalOp<NewOp, ConvertOp, ToPointersOp, ToIndicesOp, ToValuesOp,
                        ToTensorOp>();
    // All dynamic rules below accept new function, call, return, and dimop
    // operations as legal output of the rewriting provided that all sparse
    // tensor types have been fully rewritten.
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
      return converter.isSignatureLegal(op.getCalleeType());
    });
    target.addDynamicallyLegalOp<ReturnOp>(
        [&](ReturnOp op) { return converter.isLegal(op.getOperandTypes()); });
    target.addDynamicallyLegalOp<tensor::DimOp>([&](tensor::DimOp op) {
      return converter.isLegal(op.getOperandTypes());
    });
    // The following operations and dialects may be introduced by the
    // rewriting rules, and are therefore marked as legal.
    target.addLegalOp<ConstantOp, tensor::CastOp, tensor::ExtractOp>();
    target.addLegalDialect<scf::SCFDialect, LLVM::LLVMDialect,
                           memref::MemRefDialect>();
    // Populate with rules and apply rewriting rules.
    populateFuncOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateSparseTensorConversionPatterns(converter, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> mlir::createSparsificationPass() {
  return std::make_unique<SparsificationPass>();
}

std::unique_ptr<Pass> mlir::createSparseTensorConversionPass() {
  return std::make_unique<SparseTensorConversionPass>();
}
