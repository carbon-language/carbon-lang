//===- LoopAnalysis.h - loop analysis methods -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods to analyze loops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_ANALYSIS_LOOPANALYSIS_H
#define MLIR_DIALECT_AFFINE_ANALYSIS_LOOPANALYSIS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"

namespace mlir {

class AffineExpr;
class AffineForOp;
class AffineMap;
class BlockArgument;
class MemRefType;
class NestedPattern;
class Operation;
class Value;

/// Returns the trip count of the loop as an affine map with its corresponding
/// operands if the latter is expressible as an affine expression, and nullptr
/// otherwise. This method always succeeds as long as the lower bound is not a
/// multi-result map. The trip count expression is simplified before returning.
/// This method only utilizes map composition to construct lower and upper
/// bounds before computing the trip count expressions
void getTripCountMapAndOperands(AffineForOp forOp, AffineMap *map,
                                SmallVectorImpl<Value> *operands);

/// Returns the trip count of the loop if it's a constant, None otherwise. This
/// uses affine expression analysis and is able to determine constant trip count
/// in non-trivial cases.
Optional<uint64_t> getConstantTripCount(AffineForOp forOp);

/// Returns the greatest known integral divisor of the trip count. Affine
/// expression analysis is used (indirectly through getTripCount), and
/// this method is thus able to determine non-trivial divisors.
uint64_t getLargestDivisorOfTripCount(AffineForOp forOp);

/// Given an induction variable `iv` of type AffineForOp and `indices` of type
/// IndexType, returns the set of `indices` that are independent of `iv`.
///
/// Prerequisites (inherited from `isAccessInvariant` above):
///   1. `iv` and `indices` of the proper type;
///   2. at most one affine.apply is reachable from each index in `indices`;
///
/// Emits a note if it encounters a chain of affine.apply and conservatively
///  those cases.
DenseSet<Value, DenseMapInfo<Value>>
getInvariantAccesses(Value iv, ArrayRef<Value> indices);

using VectorizableLoopFun = std::function<bool(AffineForOp)>;

/// Checks whether the loop is structurally vectorizable; i.e.:
///   1. no conditionals are nested under the loop;
///   2. all nested load/stores are to scalar MemRefs.
/// TODO: relax the no-conditionals restriction
bool isVectorizableLoopBody(AffineForOp loop,
                            NestedPattern &vectorTransferMatcher);

/// Checks whether the loop is structurally vectorizable and that all the LoadOp
/// and StoreOp matched have access indexing functions that are are either:
///   1. invariant along the loop induction variable created by 'loop';
///   2. varying along at most one memory dimension. If such a unique dimension
///      is found, it is written into `memRefDim`.
bool isVectorizableLoopBody(AffineForOp loop, int *memRefDim,
                            NestedPattern &vectorTransferMatcher);

/// Checks where SSA dominance would be violated if a for op's body
/// operations are shifted by the specified shifts. This method checks if a
/// 'def' and all its uses have the same shift factor.
// TODO: extend this to check for memory-based dependence violation when we have
// the support.
bool isOpwiseShiftValid(AffineForOp forOp, ArrayRef<uint64_t> shifts);

} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_ANALYSIS_LOOPANALYSIS_H
