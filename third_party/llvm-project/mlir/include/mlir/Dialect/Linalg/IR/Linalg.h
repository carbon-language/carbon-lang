//===- Linalg.h - Linalg dialect --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_IR_LINALG_H
#define MLIR_DIALECT_LINALG_IR_LINALG_H

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace linalg {

class LinalgOp;

// TOFO: allow an extra ValueRange to specify an indexing and allow
// non-hyperrectangular shapes.
using LoopRangeBuilder =
    std::function<SmallVector<Range, 4>(ImplicitLocOpBuilder)>;

/// Provide a very simple inference procedure to build the loop ranges from the
/// op and its operands. This only works with permutation affine maps and
/// patterns of the form `(m, n)[s] -> (m + n - s floordiv 2)`.
/// A more advanced Tensor-Comprehension like inference is possible but has
/// proven to be ambiguous in unfavorable case.
/// As a consequence, we relax the default behavior very conservatively and
/// provide an op-specified hook so that Linalg ops may override the behavior.
LoopRangeBuilder defaultLoopRangesBuilder(LinalgOp op);

/// Returns the name mangled library call name to disambiguate between different
/// overloads at the C level. The name mangling scheme is basic and uses MLIR
/// type names:
///   1. form a string which is the concatenation of the linalg op name with all
///      the operand type names, separate by underscores;
///   2. drop the `linalg.` prefix, and the `<`, `>`, `?` symbols from the type.
/// Assumes `op` is a LinalgOp.
///
/// Examples:
///
/// 1. linalg.fill(%f, %A) : f32, memref<f32>
///   name mangles into `linalg_fill_f32_viewf32`
///
/// 2. linalg.dot %A, %B, %C :
///      (memref<?xf32, stride_specification>,
///       memref<?xf32, stride_specification>, memref<f32>)
///   name mangles into `linalg_dot_viewxf32_viewxf32_viewf32`
///
/// 3. linalg.matmul(...) :
///      memref<?x?xf32, stride_specification>,
///      memref<?x?xf32, stride_specification>,
///      memref<?x?xf32, stride_specification>
///   name mangles into `linalg_matmul_viewxxf32_viewxxf32_viewxxf32`
std::string generateLibraryCallName(Operation *op);

/// Returns `num` AffineDimExpr dimensions at positions
///   [startIdx, startIdx + num) and increments `startIdx` to `startIdx + num`.
SmallVector<AffineExpr, 4> makeAffineDimExprs(unsigned num, unsigned &startIdx,
                                              MLIRContext *context);

/// Returns `maybeMap.get()` if `maybeMap` is set, otherwise returns the
/// symbol-less identity map of `rank`.
AffineMap extractOrIdentityMap(Optional<AffineMap> maybeMap, unsigned rank,
                               MLIRContext *context);

/// Return the vector that is the concatenation of `a` and `b`.
SmallVector<AffineExpr, 4> concat(ArrayRef<AffineExpr> a,
                                  ArrayRef<AffineExpr> b);

/// Return the dims that are `iteratorTypeName` loops in the LinalgOp `op`.
/// Assumes `op` is a LinalgOp.
void getDimsOfType(Operation *op, StringRef iteratorTypeName,
                   SmallVectorImpl<unsigned> &res);

namespace detail {
LogicalResult verifyStructuredOpInterface(Operation *op);
} // namespace detail
} // namespace linalg
} // namespace mlir

//===----------------------------------------------------------------------===//
// Linalg Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Linalg Enums
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOpsEnums.h.inc"

//===----------------------------------------------------------------------===//
// Linalg Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOpsAttrDefs.h.inc"

//===----------------------------------------------------------------------===//
// Linalg Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

//===----------------------------------------------------------------------===//
// Linalg Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOps.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.h.inc"

#endif // MLIR_DIALECT_LINALG_IR_LINALG_H
