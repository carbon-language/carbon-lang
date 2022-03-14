//===- SPIRVToLLVMPass.cpp - SPIR-V to LLVM Passes ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR SPIR-V ops into LLVM ops
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVMPass.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

using namespace mlir;

namespace {
/// A pass converting MLIR SPIR-V operations into LLVM dialect.
class ConvertSPIRVToLLVMPass
    : public ConvertSPIRVToLLVMBase<ConvertSPIRVToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertSPIRVToLLVMPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  LLVMTypeConverter converter(&getContext());

  // Encode global variable's descriptor set and binding if they exist.
  encodeBindAttribute(module);

  RewritePatternSet patterns(context);

  populateSPIRVToLLVMTypeConversion(converter);

  populateSPIRVToLLVMModuleConversionPatterns(converter, patterns);
  populateSPIRVToLLVMConversionPatterns(converter, patterns);
  populateSPIRVToLLVMFunctionConversionPatterns(converter, patterns);

  ConversionTarget target(*context);
  target.addIllegalDialect<spirv::SPIRVDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Set `ModuleOp` as legal for `spv.module` conversion.
  target.addLegalOp<ModuleOp>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertSPIRVToLLVMPass() {
  return std::make_unique<ConvertSPIRVToLLVMPass>();
}
