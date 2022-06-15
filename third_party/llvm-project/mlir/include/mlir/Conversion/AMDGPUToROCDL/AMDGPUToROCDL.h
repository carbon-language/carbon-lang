//===- AMDGPUToROCDL.h - Convert AMDGPU to ROCDL dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_AMDGPUTOROCDL_AMDGPUTOROCDL_H_
#define MLIR_CONVERSION_AMDGPUTOROCDL_AMDGPUTOROCDL_H_

#include <memory>

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

void populateAMDGPUToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertAMDGPUToROCDLPass();

} // namespace mlir

#endif // MLIR_CONVERSION_AMDGPUTOROCDL_AMDGPUTOROCDL_H_
