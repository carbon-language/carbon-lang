//===- Utils.h - General transformation utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// memref's and non-loop IR structures. These are not passes by themselves but
// are used either by passes, optimization sequences, or in turn by other
// transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_UTILS_H
#define MLIR_TRANSFORMS_UTILS_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {

class AffineApplyOp;
class AffineForOp;
class Location;
class OpBuilder;

namespace memref {
class AllocOp;
} // end namespace memref

/// Replaces all "dereferencing" uses of `oldMemRef` with `newMemRef` while
/// optionally remapping the old memref's indices using the supplied affine map,
/// `indexRemap`. The new memref could be of a different shape or rank.
/// `extraIndices` provides any additional access indices to be added to the
/// start.
///
/// `indexRemap` remaps indices of the old memref access to a new set of indices
/// that are used to index the memref. Additional input operands to indexRemap
/// can be optionally provided in `extraOperands`, and they occupy the start
/// of its input list. `indexRemap`'s dimensional inputs are expected to
/// correspond to memref's indices, and its symbolic inputs if any should be
/// provided in `symbolOperands`.
///
/// `domOpFilter`, if non-null, restricts the replacement to only those
/// operations that are dominated by the former; similarly, `postDomOpFilter`
/// restricts replacement to only those operations that are postdominated by it.
///
/// 'allowNonDereferencingOps', if set, allows replacement of non-dereferencing
/// uses of a memref without any requirement for access index rewrites as long
/// as the user operation has the MemRefsNormalizable trait. The default value
/// of this flag is false.
///
/// 'replaceInDeallocOp', if set, lets DeallocOp, a non-dereferencing user, to
/// also be a candidate for replacement. The default value of this flag is
/// false.
///
/// Returns true on success and false if the replacement is not possible,
/// whenever a memref is used as an operand in a non-dereferencing context and
/// 'allowNonDereferencingOps' is false, except for dealloc's on the memref
/// which are left untouched. See comments at function definition for an
/// example.
//
//  Ex: to replace load %A[%i, %j] with load %Abuf[%t mod 2, %ii - %i, %j]:
//  The SSA value corresponding to '%t mod 2' should be in 'extraIndices', and
//  index remap will perform (%i, %j) -> (%ii - %i, %j), i.e., indexRemap = (d0,
//  d1, d2) -> (d0 - d1, d2), and %ii will be the extra operand. Without any
//  extra operands, note that 'indexRemap' would just be applied to existing
//  indices (%i, %j).
//  TODO: allow extraIndices to be added at any position.
LogicalResult replaceAllMemRefUsesWith(
    Value oldMemRef, Value newMemRef, ArrayRef<Value> extraIndices = {},
    AffineMap indexRemap = AffineMap(), ArrayRef<Value> extraOperands = {},
    ArrayRef<Value> symbolOperands = {}, Operation *domOpFilter = nullptr,
    Operation *postDomOpFilter = nullptr, bool allowNonDereferencingOps = false,
    bool replaceInDeallocOp = false);

/// Performs the same replacement as the other version above but only for the
/// dereferencing uses of `oldMemRef` in `op`, except in cases where
/// 'allowNonDereferencingOps' is set to true where we replace the
/// non-dereferencing uses as well.
LogicalResult replaceAllMemRefUsesWith(Value oldMemRef, Value newMemRef,
                                       Operation *op,
                                       ArrayRef<Value> extraIndices = {},
                                       AffineMap indexRemap = AffineMap(),
                                       ArrayRef<Value> extraOperands = {},
                                       ArrayRef<Value> symbolOperands = {},
                                       bool allowNonDereferencingOps = false);

/// Rewrites the memref defined by this alloc op to have an identity layout map
/// and updates all its indexing uses. Returns failure if any of its uses
/// escape (while leaving the IR in a valid state).
LogicalResult normalizeMemRef(memref::AllocOp *op);

/// Uses the old memref type map layout and computes the new memref type to have
/// a new shape and a layout map, where the old layout map has been normalized
/// to an identity layout map. It returns the old memref in case no
/// normalization was needed or a failure occurs while transforming the old map
/// layout to an identity layout map.
MemRefType normalizeMemRefType(MemRefType memrefType, OpBuilder builder,
                               unsigned numSymbolicOperands);

/// Creates and inserts into 'builder' a new AffineApplyOp, with the number of
/// its results equal to the number of operands, as a composition
/// of all other AffineApplyOps reachable from input parameter 'operands'. If
/// different operands were drawing results from multiple affine apply ops,
/// these will also be collected into a single (multi-result) affine apply op.
/// The final results of the composed AffineApplyOp are returned in output
/// parameter 'results'. Returns the affine apply op created.
Operation *createComposedAffineApplyOp(OpBuilder &builder, Location loc,
                                       ArrayRef<Value> operands,
                                       ArrayRef<Operation *> affineApplyOps,
                                       SmallVectorImpl<Value> *results);

/// Given an operation, inserts one or more single result affine apply
/// operations, results of which are exclusively used by this operation.
/// The operands of these newly created affine apply ops are
/// guaranteed to be loop iterators or terminal symbols of a function.
///
/// Before
///
/// affine.for %i = 0 to #map(%N)
///   %idx = affine.apply (d0) -> (d0 mod 2) (%i)
///   send %A[%idx], ...
///   %v = "compute"(%idx, ...)
///
/// After
///
/// affine.for %i = 0 to #map(%N)
///   %idx = affine.apply (d0) -> (d0 mod 2) (%i)
///   send %A[%idx], ...
///   %idx_ = affine.apply (d0) -> (d0 mod 2) (%i)
///   %v = "compute"(%idx_, ...)

/// This allows the application of different transformations on send and
/// compute (for eg. different shifts/delays)
///
/// Fills `sliceOps` with the list of affine.apply operations.
/// In the following cases, `sliceOps` remains empty:
///   1. If none of opInst's operands were the result of an affine.apply
///      (i.e., there was no affine computation slice to create).
///   2. If all the affine.apply op's supplying operands to this opInst did not
///      have any uses other than those in this opInst.
void createAffineComputationSlice(Operation *opInst,
                                  SmallVectorImpl<AffineApplyOp> *sliceOps);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_UTILS_H
