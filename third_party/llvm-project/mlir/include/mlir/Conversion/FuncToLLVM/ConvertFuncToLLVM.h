//===- ConvertFuncToLLVM.h - Convert Func to LLVM ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a set of conversion patterns from the Func dialect to the LLVM IR
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVM_H
#define MLIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVM_H

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

/// Collect the default pattern to convert a FuncOp to the LLVM dialect. If
/// `emitCWrappers` is set, the pattern will also produce functions
/// that pass memref descriptors by pointer-to-structure in addition to the
/// default unpacked form.
void populateFuncToLLVMFuncOpConversionPattern(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns);

/// Collect the patterns to convert from the Func dialect to LLVM. The
/// conversion patterns capture the LLVMTypeConverter and the LowerToLLVMOptions
/// by reference meaning the references have to remain alive during the entire
/// pattern lifetime.
void populateFuncToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVM_H
