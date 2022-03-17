//===- VectorToLLVM.cpp - Conversion from Vector to the LLVM dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "../PassDetail.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/AMX/Transforms.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir/Dialect/ArmSVE/Transforms.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;

namespace {
struct LowerVectorToLLVMPass
    : public ConvertVectorToLLVMBase<LowerVectorToLLVMPass> {
  LowerVectorToLLVMPass(const LowerVectorToLLVMOptions &options) {
    this->reassociateFPReductions = options.reassociateFPReductions;
    this->indexOptimizations = options.indexOptimizations;
    this->armNeon = options.armNeon;
    this->armSVE = options.armSVE;
    this->amx = options.amx;
    this->x86Vector = options.x86Vector;
  }
  // Override explicitly to allow conditional dialect dependence.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<arith::ArithmeticDialect>();
    registry.insert<memref::MemRefDialect>();
    if (armNeon)
      registry.insert<arm_neon::ArmNeonDialect>();
    if (armSVE)
      registry.insert<arm_sve::ArmSVEDialect>();
    if (amx)
      registry.insert<amx::AMXDialect>();
    if (x86Vector)
      registry.insert<x86vector::X86VectorDialect>();
  }
  void runOnOperation() override;
};
} // namespace

void LowerVectorToLLVMPass::runOnOperation() {
  // Perform progressive lowering of operations on slices and
  // all contraction operations. Also applies folding and DCE.
  {
    RewritePatternSet patterns(&getContext());
    populateVectorToVectorCanonicalizationPatterns(patterns);
    populateVectorBroadcastLoweringPatterns(patterns);
    populateVectorContractLoweringPatterns(patterns);
    populateVectorMaskOpLoweringPatterns(patterns);
    populateVectorShapeCastLoweringPatterns(patterns);
    populateVectorTransposeLoweringPatterns(patterns);
    // Vector transfer ops with rank > 1 should be lowered with VectorToSCF.
    populateVectorTransferLoweringPatterns(patterns, /*maxTransferRank=*/1);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

  // Convert to the LLVM IR dialect.
  LLVMTypeConverter converter(&getContext());
  RewritePatternSet patterns(&getContext());
  populateVectorMaskMaterializationPatterns(patterns, indexOptimizations);
  populateVectorTransferLoweringPatterns(patterns);
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns,
                                         reassociateFPReductions);
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);

  // Architecture specific augmentations.
  LLVMConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithmeticDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  if (armNeon) {
    // TODO: we may or may not want to include in-dialect lowering to
    // LLVM-compatible operations here. So far, all operations in the dialect
    // can be translated to LLVM IR so there is no conversion necessary.
    target.addLegalDialect<arm_neon::ArmNeonDialect>();
  }
  if (armSVE) {
    configureArmSVELegalizeForExportTarget(target);
    populateArmSVELegalizeForLLVMExportPatterns(converter, patterns);
  }
  if (amx) {
    configureAMXLegalizeForExportTarget(target);
    populateAMXLegalizeForLLVMExportPatterns(converter, patterns);
  }
  if (x86Vector) {
    configureX86VectorLegalizeForExportTarget(target);
    populateX86VectorLegalizeForLLVMExportPatterns(converter, patterns);
  }

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertVectorToLLVMPass(const LowerVectorToLLVMOptions &options) {
  return std::make_unique<LowerVectorToLLVMPass>(options);
}
