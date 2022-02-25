//===- LinalgTypes.h - Linalg Types ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_LINALGTYPES_H_
#define MLIR_DIALECT_LINALG_LINALGTYPES_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringMap.h"

#include "mlir/Dialect/Linalg/IR/LinalgOpsDialect.h.inc"

namespace mlir {
class MLIRContext;

namespace linalg {

/// A RangeType represents a minimal range abstraction (min, max, step).
/// It is constructed by calling the linalg.range op with three values index of
/// index type:
///
/// ```mlir
///    func @foo(%arg0 : index, %arg1 : index, %arg2 : index) {
///      %0 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
///    }
/// ```
class RangeType : public Type::TypeBase<RangeType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_LINALGTYPES_H_
