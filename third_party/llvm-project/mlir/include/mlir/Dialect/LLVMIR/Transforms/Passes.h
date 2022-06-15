//===- Passes.h - LLVM Pass Construction and Registration -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES_H

#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/Dialect/LLVMIR/Transforms/OptimizeForNVVM.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace LLVM {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES_H
