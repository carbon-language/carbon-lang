//===- VectorUtils.cpp - MLIR Utilities for VectorOps   ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility methods for working with the Vector dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Utils/VectorUtils.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include <numeric>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

/// Helper function that creates a memref::DimOp or tensor::DimOp depending on
/// the type of `source`.
Value mlir::vector::createOrFoldDimOp(OpBuilder &b, Location loc, Value source,
                                      int64_t dim) {
  if (source.getType().isa<UnrankedMemRefType, MemRefType>())
    return b.createOrFold<memref::DimOp>(loc, source, dim);
  if (source.getType().isa<UnrankedTensorType, RankedTensorType>())
    return b.createOrFold<tensor::DimOp>(loc, source, dim);
  llvm_unreachable("Expected MemRefType or TensorType");
}

Value mlir::vector::makeArithReduction(OpBuilder &b, Location loc,
                                       CombiningKind kind, Value v1, Value v2) {
  Type t1 = getElementTypeOrSelf(v1.getType());
  Type t2 = getElementTypeOrSelf(v2.getType());
  switch (kind) {
  case CombiningKind::ADD:
    if (t1.isIntOrIndex() && t2.isIntOrIndex())
      return b.createOrFold<arith::AddIOp>(loc, v1, v2);
    else if (t1.isa<FloatType>() && t2.isa<FloatType>())
      return b.createOrFold<arith::AddFOp>(loc, v1, v2);
    llvm_unreachable("invalid value types for ADD reduction");
  case CombiningKind::AND:
    assert(t1.isIntOrIndex() && t2.isIntOrIndex() && "expected int values");
    return b.createOrFold<arith::AndIOp>(loc, v1, v2);
  case CombiningKind::MAXF:
    assert(t1.isa<FloatType>() && t2.isa<FloatType>() &&
           "expected float values");
    return b.createOrFold<arith::MaxFOp>(loc, v1, v2);
  case CombiningKind::MINF:
    assert(t1.isa<FloatType>() && t2.isa<FloatType>() &&
           "expected float values");
    return b.createOrFold<arith::MinFOp>(loc, v1, v2);
  case CombiningKind::MAXSI:
    assert(t1.isIntOrIndex() && t2.isIntOrIndex() && "expected int values");
    return b.createOrFold<arith::MaxSIOp>(loc, v1, v2);
  case CombiningKind::MINSI:
    assert(t1.isIntOrIndex() && t2.isIntOrIndex() && "expected int values");
    return b.createOrFold<arith::MinSIOp>(loc, v1, v2);
  case CombiningKind::MAXUI:
    assert(t1.isIntOrIndex() && t2.isIntOrIndex() && "expected int values");
    return b.createOrFold<arith::MaxUIOp>(loc, v1, v2);
  case CombiningKind::MINUI:
    assert(t1.isIntOrIndex() && t2.isIntOrIndex() && "expected int values");
    return b.createOrFold<arith::MinUIOp>(loc, v1, v2);
  case CombiningKind::MUL:
    if (t1.isIntOrIndex() && t2.isIntOrIndex())
      return b.createOrFold<arith::MulIOp>(loc, v1, v2);
    else if (t1.isa<FloatType>() && t2.isa<FloatType>())
      return b.createOrFold<arith::MulFOp>(loc, v1, v2);
    llvm_unreachable("invalid value types for MUL reduction");
  case CombiningKind::OR:
    assert(t1.isIntOrIndex() && t2.isIntOrIndex() && "expected int values");
    return b.createOrFold<arith::OrIOp>(loc, v1, v2);
  case CombiningKind::XOR:
    assert(t1.isIntOrIndex() && t2.isIntOrIndex() && "expected int values");
    return b.createOrFold<arith::XOrIOp>(loc, v1, v2);
  };
  llvm_unreachable("unknown CombiningKind");
}

/// Return the number of elements of basis, `0` if empty.
int64_t mlir::computeMaxLinearIndex(ArrayRef<int64_t> basis) {
  if (basis.empty())
    return 0;
  return std::accumulate(basis.begin(), basis.end(), 1,
                         std::multiplies<int64_t>());
}

SmallVector<int64_t, 4> mlir::computeStrides(ArrayRef<int64_t> shape,
                                             ArrayRef<int64_t> sizes) {
  int64_t rank = shape.size();
  // Compute the count for each dimension.
  SmallVector<int64_t, 4> sliceDimCounts(rank);
  for (int64_t r = 0; r < rank; ++r)
    sliceDimCounts[r] = ceilDiv(shape[r], sizes[r]);
  // Use that to compute the slice stride for each dimension.
  SmallVector<int64_t, 4> sliceStrides(rank);
  sliceStrides[rank - 1] = 1;
  for (int64_t r = rank - 2; r >= 0; --r)
    sliceStrides[r] = sliceStrides[r + 1] * sliceDimCounts[r + 1];
  return sliceStrides;
}

SmallVector<int64_t, 4> mlir::computeElementOffsetsFromVectorSliceOffsets(
    ArrayRef<int64_t> sizes, ArrayRef<int64_t> vectorOffsets) {
  SmallVector<int64_t, 4> result;
  for (auto it : llvm::zip(vectorOffsets, sizes))
    result.push_back(std::get<0>(it) * std::get<1>(it));
  return result;
}

Optional<SmallVector<int64_t, 4>> mlir::shapeRatio(ArrayRef<int64_t> superShape,
                                                   ArrayRef<int64_t> subShape) {
  if (superShape.size() < subShape.size()) {
    return Optional<SmallVector<int64_t, 4>>();
  }

  // Starting from the end, compute the integer divisors.
  std::vector<int64_t> result;
  result.reserve(superShape.size());
  int64_t superSize = 0, subSize = 0;
  for (auto it :
       llvm::zip(llvm::reverse(superShape), llvm::reverse(subShape))) {
    std::tie(superSize, subSize) = it;
    assert(superSize > 0 && "superSize must be > 0");
    assert(subSize > 0 && "subSize must be > 0");

    // If integral division does not occur, return and let the caller decide.
    if (superSize % subSize != 0)
      return None;
    result.push_back(superSize / subSize);
  }

  // At this point we computed the ratio (in reverse) for the common
  // size. Fill with the remaining entries from the super-vector shape (still in
  // reverse).
  int commonSize = subShape.size();
  std::copy(superShape.rbegin() + commonSize, superShape.rend(),
            std::back_inserter(result));

  assert(result.size() == superShape.size() &&
         "super to sub shape ratio is not of the same size as the super rank");

  // Reverse again to get it back in the proper order and return.
  return SmallVector<int64_t, 4>{result.rbegin(), result.rend()};
}

Optional<SmallVector<int64_t, 4>> mlir::shapeRatio(VectorType superVectorType,
                                                   VectorType subVectorType) {
  assert(superVectorType.getElementType() == subVectorType.getElementType() &&
         "vector types must be of the same elemental type");
  return shapeRatio(superVectorType.getShape(), subVectorType.getShape());
}

/// Constructs a permutation map from memref indices to vector dimension.
///
/// The implementation uses the knowledge of the mapping of enclosing loop to
/// vector dimension. `enclosingLoopToVectorDim` carries this information as a
/// map with:
///   - keys representing "vectorized enclosing loops";
///   - values representing the corresponding vector dimension.
/// The algorithm traverses "vectorized enclosing loops" and extracts the
/// at-most-one MemRef index that is invariant along said loop. This index is
/// guaranteed to be at most one by construction: otherwise the MemRef is not
/// vectorizable.
/// If this invariant index is found, it is added to the permutation_map at the
/// proper vector dimension.
/// If no index is found to be invariant, 0 is added to the permutation_map and
/// corresponds to a vector broadcast along that dimension.
///
/// Returns an empty AffineMap if `enclosingLoopToVectorDim` is empty,
/// signalling that no permutation map can be constructed given
/// `enclosingLoopToVectorDim`.
///
/// Examples can be found in the documentation of `makePermutationMap`, in the
/// header file.
static AffineMap makePermutationMap(
    ArrayRef<Value> indices,
    const DenseMap<Operation *, unsigned> &enclosingLoopToVectorDim) {
  if (enclosingLoopToVectorDim.empty())
    return AffineMap();
  MLIRContext *context =
      enclosingLoopToVectorDim.begin()->getFirst()->getContext();
  SmallVector<AffineExpr, 4> perm(enclosingLoopToVectorDim.size(),
                                  getAffineConstantExpr(0, context));

  for (auto kvp : enclosingLoopToVectorDim) {
    assert(kvp.second < perm.size());
    auto invariants = getInvariantAccesses(
        cast<AffineForOp>(kvp.first).getInductionVar(), indices);
    unsigned numIndices = indices.size();
    unsigned countInvariantIndices = 0;
    for (unsigned dim = 0; dim < numIndices; ++dim) {
      if (!invariants.count(indices[dim])) {
        assert(perm[kvp.second] == getAffineConstantExpr(0, context) &&
               "permutationMap already has an entry along dim");
        perm[kvp.second] = getAffineDimExpr(dim, context);
      } else {
        ++countInvariantIndices;
      }
    }
    assert((countInvariantIndices == numIndices ||
            countInvariantIndices == numIndices - 1) &&
           "Vectorization prerequisite violated: at most 1 index may be "
           "invariant wrt a vectorized loop");
  }
  return AffineMap::get(indices.size(), 0, perm, context);
}

/// Implementation detail that walks up the parents and records the ones with
/// the specified type.
/// TODO: could also be implemented as a collect parents followed by a
/// filter and made available outside this file.
template <typename T>
static SetVector<Operation *> getParentsOfType(Block *block) {
  SetVector<Operation *> res;
  auto *current = block->getParentOp();
  while (current) {
    if (auto typedParent = dyn_cast<T>(current)) {
      assert(res.count(current) == 0 && "Already inserted");
      res.insert(current);
    }
    current = current->getParentOp();
  }
  return res;
}

/// Returns the enclosing AffineForOp, from closest to farthest.
static SetVector<Operation *> getEnclosingforOps(Block *block) {
  return getParentsOfType<AffineForOp>(block);
}

AffineMap mlir::makePermutationMap(
    Block *insertPoint, ArrayRef<Value> indices,
    const DenseMap<Operation *, unsigned> &loopToVectorDim) {
  DenseMap<Operation *, unsigned> enclosingLoopToVectorDim;
  auto enclosingLoops = getEnclosingforOps(insertPoint);
  for (auto *forInst : enclosingLoops) {
    auto it = loopToVectorDim.find(forInst);
    if (it != loopToVectorDim.end()) {
      enclosingLoopToVectorDim.insert(*it);
    }
  }
  return ::makePermutationMap(indices, enclosingLoopToVectorDim);
}

AffineMap mlir::makePermutationMap(
    Operation *op, ArrayRef<Value> indices,
    const DenseMap<Operation *, unsigned> &loopToVectorDim) {
  return makePermutationMap(op->getBlock(), indices, loopToVectorDim);
}

bool matcher::operatesOnSuperVectorsOf(Operation &op,
                                       VectorType subVectorType) {
  // First, extract the vector type and distinguish between:
  //   a. ops that *must* lower a super-vector (i.e. vector.transfer_read,
  //      vector.transfer_write); and
  //   b. ops that *may* lower a super-vector (all other ops).
  // The ops that *may* lower a super-vector only do so if the super-vector to
  // sub-vector ratio exists. The ops that *must* lower a super-vector are
  // explicitly checked for this property.
  /// TODO: there should be a single function for all ops to do this so we
  /// do not have to special case. Maybe a trait, or just a method, unclear atm.
  bool mustDivide = false;
  (void)mustDivide;
  VectorType superVectorType;
  if (auto transfer = dyn_cast<VectorTransferOpInterface>(op)) {
    superVectorType = transfer.getVectorType();
    mustDivide = true;
  } else if (op.getNumResults() == 0) {
    if (!isa<func::ReturnOp>(op)) {
      op.emitError("NYI: assuming only return operations can have 0 "
                   " results at this point");
    }
    return false;
  } else if (op.getNumResults() == 1) {
    if (auto v = op.getResult(0).getType().dyn_cast<VectorType>()) {
      superVectorType = v;
    } else {
      // Not a vector type.
      return false;
    }
  } else {
    // Not a vector.transfer and has more than 1 result, fail hard for now to
    // wake us up when something changes.
    op.emitError("NYI: operation has more than 1 result");
    return false;
  }

  // Get the ratio.
  auto ratio = shapeRatio(superVectorType, subVectorType);

  // Sanity check.
  assert((ratio.hasValue() || !mustDivide) &&
         "vector.transfer operation in which super-vector size is not an"
         " integer multiple of sub-vector size");

  // This catches cases that are not strictly necessary to have multiplicity but
  // still aren't divisible by the sub-vector shape.
  // This could be useful information if we wanted to reshape at the level of
  // the vector type (but we would have to look at the compute and distinguish
  // between parallel, reduction and possibly other cases.
  return ratio.hasValue();
}
