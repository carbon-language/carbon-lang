//===- SPIRVGLSLCanonicalization.cpp - SPIR-V GLSL canonicalization patterns =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the canonicalization patterns for SPIR-V GLSL-specific ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVGLSLCanonicalization.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

using namespace mlir;

namespace {
#include "SPIRVCanonicalization.inc"
} // end anonymous namespace

namespace mlir {
namespace spirv {
void populateSPIRVGLSLCanonicalizationPatterns(RewritePatternSet &results) {
  results.add<ConvertComparisonIntoClampSPV_FOrdLessThanOp,
              ConvertComparisonIntoClampSPV_FOrdLessThanEqualOp,
              ConvertComparisonIntoClampSPV_SLessThanOp,
              ConvertComparisonIntoClampSPV_SLessThanEqualOp,
              ConvertComparisonIntoClampSPV_ULessThanOp,
              ConvertComparisonIntoClampSPV_ULessThanEqualOp>(
      results.getContext());
}
} // namespace spirv
} // namespace mlir
