//===- Async.h - MLIR Async dialect -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the async dialect that is used for modeling asynchronous
// execution.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ASYNC_IR_ASYNC_H
#define MLIR_DIALECT_ASYNC_IR_ASYNC_H

#include "mlir/Dialect/Async/IR/AsyncTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Async Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Async/IR/AsyncOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Async Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Async/IR/AsyncOps.h.inc"

//===----------------------------------------------------------------------===//
// Helper functions of Async dialect transformations.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace async {

/// Returns true if the type is reference counted at runtime.
inline bool isRefCounted(Type type) {
  return type.isa<TokenType, ValueType, GroupType>();
}

} // namespace async
} // namespace mlir

#endif // MLIR_DIALECT_ASYNC_IR_ASYNC_H
