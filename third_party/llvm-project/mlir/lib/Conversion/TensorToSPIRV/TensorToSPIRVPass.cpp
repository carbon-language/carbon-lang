//===- TensorToSPIRVPass.cpp - Tensor to SPIR-V Passes ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert Tensor dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TensorToSPIRV/TensorToSPIRVPass.h"
#include "../PassDetail.h"
#include "mlir/Conversion/ArithmeticToSPIRV/ArithmeticToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/TensorToSPIRV/TensorToSPIRV.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

using namespace mlir;

namespace {
/// A pass converting MLIR Tensor operations into the SPIR-V dialect.
class ConvertTensorToSPIRVPass
    : public ConvertTensorToSPIRVBase<ConvertTensorToSPIRVPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    auto targetAttr = spirv::lookupTargetEnvOrDefault(module);
    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    SPIRVTypeConverter::Options options;
    options.emulateNon32BitScalarTypes = this->emulateNon32BitScalarTypes;
    SPIRVTypeConverter typeConverter(targetAttr, options);

    RewritePatternSet patterns(context);
    arith::populateArithmeticToSPIRVPatterns(typeConverter, patterns);
    populateFuncToSPIRVPatterns(typeConverter, patterns);
    populateTensorToSPIRVPatterns(typeConverter, /*byteCountThreshold=*/64,
                                  patterns);
    populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, *target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertTensorToSPIRVPass() {
  return std::make_unique<ConvertTensorToSPIRVPass>();
}
