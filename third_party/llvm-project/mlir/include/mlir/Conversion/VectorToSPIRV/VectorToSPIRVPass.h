//=- VectorToSPIRVPass.h - Vector to SPIR-V Passes ----------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert Vector dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_VECTORTOSPIRV_GPUTOSPIRVPASS_H
#define MLIR_CONVERSION_VECTORTOSPIRV_GPUTOSPIRVPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Creates a pass to convert Vector Ops to SPIR-V ops.
std::unique_ptr<OperationPass<ModuleOp>> createConvertVectorToSPIRVPass();

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOSPIRV_GPUTOSPIRVPASS_H
