//===- TestDialect.h - MLIR Dialect for testing -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a fake 'test' dialect that can be used for testing things
// that do not have a respective counterpart in the main source directories.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTDIALECT_H
#define MLIR_TESTDIALECT_H

#include "TestInterfaces.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class DLTIDialect;
class RewritePatternSet;
} // namespace mlir

#include "TestOpEnums.h.inc"
#include "TestOpInterfaces.h.inc"
#include "TestOpStructs.h.inc"
#include "TestOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TestOps.h.inc"

namespace test {
void registerTestDialect(::mlir::DialectRegistry &registry);
void populateTestReductionPatterns(::mlir::RewritePatternSet &patterns);
} // end namespace test

#endif // MLIR_TESTDIALECT_H
