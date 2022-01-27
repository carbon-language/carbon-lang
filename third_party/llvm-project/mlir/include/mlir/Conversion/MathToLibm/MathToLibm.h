//===- MathToLibm.h - Utils to convert from the complex dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_MATHTOLIBM_MATHTOLIBM_H_
#define MLIR_CONVERSION_MATHTOLIBM_MATHTOLIBM_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
template <typename T>
class OperationPass;

/// Populate the given list with patterns that convert from Math to Libm calls.
void populateMathToLibmConversionPatterns(RewritePatternSet &patterns,
                                          PatternBenefit benefit);

/// Create a pass to convert Math operations to libm calls.
std::unique_ptr<OperationPass<ModuleOp>> createConvertMathToLibmPass();

} // namespace mlir

#endif // MLIR_CONVERSION_MATHTOLIBM_MATHTOLIBM_H_
