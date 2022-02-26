//===- Utils.h - SCF dialect utilities --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various SCF utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_UTILS_UTILS_H_
#define MLIR_DIALECT_SCF_UTILS_UTILS_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
class FuncOp;
class Location;
class Operation;
class OpBuilder;
class Region;
class RewriterBase;
class ValueRange;
class Value;

namespace scf {
class IfOp;
class ForOp;
class ParallelOp;
} // namespace scf

/// Create a clone of `loop` with `newIterOperands` added as new initialization
/// values and `newYieldedValues` added as new yielded values. The returned
/// ForOp has `newYieldedValues.size()` new result values.  The `loop` induction
/// variable and `newIterOperands` are remapped to the new induction variable
/// and the new entry block arguments respectively.
///
/// Additionally, if `replaceLoopResults` is true, all uses of
/// `loop.getResults()` are replaced with the first `loop.getNumResults()`
/// return values respectively. This additional replacement is provided as a
/// convenience to update the consumers of `loop`, in the case e.g. when `loop`
/// is soon to be deleted.
///
/// Return the cloned loop.
///
/// This convenience function is useful to factorize common mechanisms related
/// to hoisting roundtrips to memory into yields. It does not perform any
/// legality checks.
///
/// Prerequisite: `newYieldedValues.size() == newYieldedValues.size()`.
scf::ForOp cloneWithNewYields(OpBuilder &b, scf::ForOp loop,
                              ValueRange newIterOperands,
                              ValueRange newYieldedValues,
                              bool replaceLoopResults = true);

/// Outline a region with a single block into a new FuncOp.
/// Assumes the FuncOp result types is the type of the yielded operands of the
/// single block. This constraint makes it easy to determine the result.
/// This method also clones the `arith::ConstantIndexOp` at the start of
/// `outlinedFuncBody` to alloc simple canonicalizations.
/// Creates a new FuncOp and thus cannot be used in a FuncOp pass.
/// The client is responsible for providing a unique `funcName` that will not
/// collide with another FuncOp name.
// TODO: support more than single-block regions.
// TODO: more flexible constant handling.
FailureOr<FuncOp> outlineSingleBlockRegion(RewriterBase &rewriter, Location loc,
                                           Region &region, StringRef funcName);

/// Outline the then and/or else regions of `ifOp` as follows:
///  - if `thenFn` is not null, `thenFnName` must be specified and the `then`
///    region is inlined into a new FuncOp that is captured by the pointer.
///  - if `elseFn` is not null, `elseFnName` must be specified and the `else`
///    region is inlined into a new FuncOp that is captured by the pointer.
/// Creates new FuncOps and thus cannot be used in a FuncOp pass.
/// The client is responsible for providing a unique `thenFnName`/`elseFnName`
/// that will not collide with another FuncOp name.
LogicalResult outlineIfOp(RewriterBase &b, scf::IfOp ifOp, FuncOp *thenFn,
                          StringRef thenFnName, FuncOp *elseFn,
                          StringRef elseFnName);

/// Get a list of innermost parallel loops contained in `rootOp`. Innermost
/// parallel loops are those that do not contain further parallel loops
/// themselves.
bool getInnermostParallelLoops(Operation *rootOp,
                               SmallVectorImpl<scf::ParallelOp> &result);

/// Return the min/max expressions for `value` if it is an induction variable
/// from scf.for or scf.parallel loop.
/// if `loopFilter` is passed, the filter determines which loop to consider.
/// Other induction variables are ignored.
Optional<std::pair<AffineExpr, AffineExpr>>
getSCFMinMaxExpr(Value value, SmallVectorImpl<Value> &dims,
                 SmallVectorImpl<Value> &symbols,
                 llvm::function_ref<bool(Operation *)> loopFilter = nullptr);

/// Replace a perfect nest of "for" loops with a single linearized loop. Assumes
/// `loops` contains a list of perfectly nested loops with bounds and steps
/// independent of any loop induction variable involved in the nest.
void coalesceLoops(MutableArrayRef<scf::ForOp> loops);

/// Take the ParallelLoop and for each set of dimension indices, combine them
/// into a single dimension. combinedDimensions must contain each index into
/// loops exactly once.
void collapseParallelLoops(scf::ParallelOp loops,
                           ArrayRef<std::vector<unsigned>> combinedDimensions);

/// Promotes the loop body of a scf::ForOp to its containing block if the loop
/// was known to have a single iteration.
LogicalResult promoteIfSingleIteration(scf::ForOp forOp);

/// Unrolls this for operation by the specified unroll factor. Returns failure
/// if the loop cannot be unrolled either due to restrictions or due to invalid
/// unroll factors. Requires positive loop bounds and step. If specified,
/// annotates the Ops in each unrolled iteration by applying `annotateFn`.
LogicalResult loopUnrollByFactor(
    scf::ForOp forOp, uint64_t unrollFactor,
    function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn = nullptr);

/// Tile a nest of standard for loops rooted at `rootForOp` by finding such
/// parametric tile sizes that the outer loops have a fixed number of iterations
/// as defined in `sizes`.
using Loops = SmallVector<scf::ForOp, 8>;
using TileLoops = std::pair<Loops, Loops>;
TileLoops extractFixedOuterLoops(scf::ForOp rootFOrOp, ArrayRef<int64_t> sizes);

/// Performs tiling fo imperfectly nested loops (with interchange) by
/// strip-mining the `forOps` by `sizes` and sinking them, in their order of
/// occurrence in `forOps`, under each of the `targets`.
/// Returns the new AffineForOps, one per each of (`forOps`, `targets`) pair,
/// nested immediately under each of `targets`.
SmallVector<Loops, 8> tile(ArrayRef<scf::ForOp> forOps, ArrayRef<Value> sizes,
                           ArrayRef<scf::ForOp> targets);

/// Performs tiling (with interchange) by strip-mining the `forOps` by `sizes`
/// and sinking them, in their order of occurrence in `forOps`, under `target`.
/// Returns the new AffineForOps, one per `forOps`, nested immediately under
/// `target`.
Loops tile(ArrayRef<scf::ForOp> forOps, ArrayRef<Value> sizes,
           scf::ForOp target);

/// Tile a nest of scf::ForOp loops rooted at `rootForOp` with the given
/// (parametric) sizes. Sizes are expected to be strictly positive values at
/// runtime.  If more sizes than loops are provided, discard the trailing values
/// in sizes.  Assumes the loop nest is permutable.
/// Returns the newly created intra-tile loops.
Loops tilePerfectlyNested(scf::ForOp rootForOp, ArrayRef<Value> sizes);

/// Get perfectly nested sequence of loops starting at root of loop nest
/// (the first op being another AffineFor, and the second op - a terminator).
/// A loop is perfectly nested iff: the first op in the loop's body is another
/// AffineForOp, and the second op is a terminator).
void getPerfectlyNestedLoops(SmallVectorImpl<scf::ForOp> &nestedLoops,
                             scf::ForOp root);

} // namespace mlir

#endif // MLIR_DIALECT_SCF_UTILS_UTILS_H_
