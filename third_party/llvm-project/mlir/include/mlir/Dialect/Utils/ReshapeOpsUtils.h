//===- ReshapeOpsUtils.h - Utilities used by reshape ops --*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities and common canonicalization patterns for
// reshape operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UTILS_RESHAPEOPSUTILS_H
#define MLIR_DIALECT_UTILS_RESHAPEOPSUTILS_H

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

using ReassociationIndices = SmallVector<int64_t, 2>;
using ReassociationIndicesRef = ArrayRef<int64_t>;
using ReassociationExprs = SmallVector<AffineExpr, 2>;

/// Attribute name for the ArrayAttr which encodes reassociation indices.
constexpr StringRef getReassociationAttrName() { return "reassociation"; }

/// Compose reassociation maps that are used in pair of reshape ops where one
/// is a producer and other is the consumer. Only valid to use this method when
/// both the producer and consumer are collapsing dimensions or both are
/// expanding dimensions.
///
/// For example,
///   producerReassociation = [[0, 1], [2], [3, 4]]
///   consumerReassociation = [[0, 1], [2]]
///
/// is folded into
///
///   result = [[0, 1, 2], [3, 4]].
Optional<SmallVector<ReassociationIndices>> composeReassociationIndices(
    ArrayRef<ReassociationIndices> producerReassociations,
    ArrayRef<ReassociationIndices> consumerReassociations,
    MLIRContext *context);

/// Convert reassociation indices to affine expressions.
SmallVector<SmallVector<AffineExpr, 2>, 2> convertReassociationIndicesToExprs(
    MLIRContext *context, ArrayRef<ReassociationIndices> reassociationIndices);

/// Constructs affine maps out of Array<Array<AffineExpr>>.
SmallVector<AffineMap, 4>
getSymbolLessAffineMaps(ArrayRef<ReassociationExprs> reassociation);

/// Wraps a list of reassociations in an ArrayAttr.
ArrayAttr
getReassociationIndicesAttribute(OpBuilder &b,
                                 ArrayRef<ReassociationIndices> reassociation);

/// Convert Array<Array<AffineExpr>> to Array<Array<int64_t>>.
SmallVector<ReassociationIndices, 2> convertReassociationMapsToIndices(
    OpBuilder &b, ArrayRef<ReassociationExprs> reassociationExprs);

/// Return the reassociations maps to use to reshape given the source type and
/// the target type when possible. Return llvm::None when this computation
/// failed.
Optional<SmallVector<ReassociationIndices>>
getReassociationIndicesForReshape(ShapedType sourceType, ShapedType targetType);

/// Returns the reassociation maps to collapse `sourceShape` to `targetShape` if
/// possible.
Optional<SmallVector<ReassociationIndices>>
getReassociationIndicesForCollapse(ArrayRef<int64_t> sourceShape,
                                   ArrayRef<int64_t> targetShape);

/// Return true if the reassociation specification is valid, false otherwise.
/// When false, the `invalidIndex` integer pointer is optionally filled with the
/// index of the offending reassociation map.
bool isReassociationValid(ArrayRef<AffineMap> reassociation,
                          int *invalidIndex = nullptr);

template <typename ReshapeOpTy, typename InverseReshapeOpTy>
static OpFoldResult foldReshapeOp(ReshapeOpTy reshapeOp,
                                  ArrayRef<Attribute> operands) {
  // Fold producer-consumer reshape ops that where the operand type of the
  // producer is same as the return type of the consumer.
  auto reshapeSrcOp =
      reshapeOp.src().template getDefiningOp<InverseReshapeOpTy>();
  if (reshapeSrcOp && reshapeSrcOp.getSrcType() == reshapeOp.getResultType())
    return reshapeSrcOp.src();
  // Reshape of a constant can be replaced with a new constant.
  if (auto elements = operands.front().dyn_cast_or_null<DenseElementsAttr>()) {
    return elements.reshape(
        reshapeOp.getResult().getType().template cast<ShapedType>());
  }
  return nullptr;
}

/// Common verifier for reshape-like types. Fills `expandedType` and
///`collapsedType` with the proper `src` or `result` type.
template <typename Op, typename T>
static LogicalResult verifyReshapeLikeTypes(Op op, T expandedType,
                                            T collapsedType, bool isExpansion) {
  unsigned expandedRank = expandedType.getRank();
  unsigned collapsedRank = collapsedType.getRank();
  if (expandedRank < collapsedRank)
    return op.emitOpError("expected the type ")
           << expandedType
           << " to have higher rank than the type = " << collapsedType;
  if (expandedRank == 0)
    return op.emitOpError("expected non-zero memref ranks");
  if (expandedRank == collapsedRank)
    return op.emitOpError("expected to collapse or expand dims");

  if (collapsedRank == 0) {
    // If collapsed rank is 0, then expanded type must be static shaped and of
    // sizes 1.
    if (llvm::any_of(expandedType.getShape(),
                     [](int64_t dim) -> bool { return dim != 1; }))
      return op.emitOpError("invalid to reshape tensor/memref with non-unit "
                            "extent dimensions to zero-rank tensor/memref");
    return success();
  }
  if (collapsedRank != op.reassociation().size())
    return op.emitOpError("expected rank of the collapsed type(")
           << collapsedRank << ") to be the number of reassociation maps("
           << op.reassociation().size() << ")";
  auto maps = op.getReassociationMaps();
  for (auto it : llvm::enumerate(maps))
    if (it.value().getNumDims() != expandedRank)
      return op.emitOpError("expected reassociation map #")
             << it.index() << " of same rank as expanded memref("
             << expandedRank << "), but got " << it.value().getNumDims();
  int invalidIdx = 0;
  if (!isReassociationValid(maps, &invalidIdx))
    return op.emitOpError("expected reassociation map #")
           << invalidIdx << " to be valid and contiguous";
  return verifyReshapeLikeShapes(op, collapsedType, expandedType, isExpansion);
}

/// Verify that shapes of the reshaped types using following rules
/// 1) if a dimension in the collapsed type is static, then the corresponding
///    dimensions in the expanded shape should be
///    a) static
///    b) the product should be same as the collaped shape.
/// 2) if a dimension in the collaped type is dynamic, one and only one of the
///    corresponding dimensions in the expanded type should be dynamic. This
///    rule is only needed with reshape operations that are expanding.
LogicalResult reshapeLikeShapesAreCompatible(
    function_ref<LogicalResult(const Twine &)> emitError,
    ArrayRef<int64_t> collapsedShape, ArrayRef<int64_t> expandedShape,
    ArrayRef<ReassociationIndices> reassociationMaps, bool isExpandingReshape);

template <typename OpTy>
static LogicalResult verifyReshapeLikeShapes(OpTy op, ShapedType collapsedType,
                                             ShapedType expandedType,
                                             bool isExpandingReshape) {
  return reshapeLikeShapesAreCompatible(
      [&](const Twine &msg) { return op->emitOpError(msg); },
      collapsedType.getShape(), expandedType.getShape(),
      op.getReassociationIndices(), isExpandingReshape);
}

/// Returns true iff the type is a MemRefType and has a non-identity layout.
bool hasNonIdentityLayout(Type type);

/// Pattern to collapse producer/consumer reshape ops that are both collapsing
/// dimensions or are both expanding dimensions.
template <typename ReshapeOpTy>
struct ComposeReassociativeReshapeOps : public OpRewritePattern<ReshapeOpTy> {
  using OpRewritePattern<ReshapeOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOpTy reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto srcReshapeOp = reshapeOp.src().template getDefiningOp<ReshapeOpTy>();
    if (!srcReshapeOp)
      return failure();

    ShapedType resultType = reshapeOp.getResultType();

    if (hasNonIdentityLayout(srcReshapeOp.src().getType()) ||
        hasNonIdentityLayout(reshapeOp.src().getType()) ||
        hasNonIdentityLayout(reshapeOp.result().getType()))
      return failure();

    Optional<SmallVector<ReassociationIndices>> reassociationIndices =
        composeReassociationIndices(srcReshapeOp.getReassociationIndices(),
                                    reshapeOp.getReassociationIndices(),
                                    rewriter.getContext());
    if (!reassociationIndices)
      return failure();
    rewriter.replaceOpWithNewOp<ReshapeOpTy>(
        reshapeOp, resultType, srcReshapeOp.src(), *reassociationIndices);
    return success();
  }
};

/// Pattern to compose
/// `collapse_shape(expand_shape(%src, reassociation_1), reassociation_2)`.
/// In that case both `srcType` and `resultType` can be expressed as a function
/// of `intermediateType`.
/// In order to demonstrate the approach, let's assume that `rank(srcType) >
/// `rank(resultType)`, i.e. the resulting operation should be `collapse_shape`.
/// In that case, we can iterate over every set of indices in `reassociation_2`
/// and try to find ids of sets of indices in `reassociation_1` that cover it
/// completely.
///
/// Example:
///
///   %0 = tensor.expand_shape %arg [[0], [1], [2, 3]]
///     : tensor<?x?x?xi64> into tensor<?x?x?x1xi64>
///   %1 = tensor.collapse_shape %0 [[0, 1], [2, 3]]
///     : tensor<?x?x?x1xi64> into tensor<?x?xi64>
///
/// can be canonicalized into
///
///   %0 = tensor.collapse_shape %arg [[0, 1], [2]]
///     : tensor<?x?x?xi64> into tensor<?x?xi64>
///
/// because [0] and [1] from `expand_shape` reassociation cover completely
/// `[0, 1]` from `collapse_shape`. If it is impossible to find such union of
/// indices, then we fail.
//
/// When `rank(srcType) < rank(resultType)`, then we just swap `reassociation_1`
/// `reassociation_2` and produce `expand_shape`.
template <typename CollapseOpTy, typename ExpandOpTy>
struct ComposeCollapseOfExpandOp : public OpRewritePattern<CollapseOpTy> {
  using OpRewritePattern<CollapseOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(CollapseOpTy collapseOp,
                                PatternRewriter &rewriter) const override {
    auto expandOp = collapseOp.src().template getDefiningOp<ExpandOpTy>();
    if (!expandOp)
      return failure();

    ShapedType srcType = expandOp.getSrcType();
    ShapedType resultType = collapseOp.getResultType();

    if (hasNonIdentityLayout(collapseOp.src().getType()) ||
        hasNonIdentityLayout(expandOp.src().getType()) ||
        hasNonIdentityLayout(expandOp.result().getType()))
      return failure();

    int64_t srcRank = srcType.getRank();
    int64_t resultRank = resultType.getRank();
    if (srcType == resultType)
      return failure();

    SmallVector<ReassociationIndices, 4> higherRankReassociation,
        lowerRankReassociation;

    bool isResultCollapsed = srcRank > resultRank;
    if (isResultCollapsed) {
      higherRankReassociation = expandOp.getReassociationIndices();
      lowerRankReassociation = collapseOp.getReassociationIndices();
    } else {
      higherRankReassociation = collapseOp.getReassociationIndices();
      lowerRankReassociation = expandOp.getReassociationIndices();
    }

    size_t higherRankIndicesID = 0;
    SmallVector<ReassociationIndices, 4> composedReassociation;
    for (const auto &lowerRankIndices : lowerRankReassociation) {
      ReassociationIndices composedIndices;
      while (higherRankIndicesID < higherRankReassociation.size()) {
        auto rightmostIndex =
            higherRankReassociation[higherRankIndicesID].back();
        if (rightmostIndex > lowerRankIndices.back())
          return failure();
        composedIndices.push_back(higherRankIndicesID++);
        if (rightmostIndex == lowerRankIndices.back())
          break;
      }
      composedReassociation.push_back(composedIndices);
    }
    if (isResultCollapsed)
      rewriter.replaceOpWithNewOp<CollapseOpTy>(
          collapseOp, resultType, expandOp.src(), composedReassociation);
    else
      rewriter.replaceOpWithNewOp<ExpandOpTy>(
          collapseOp, resultType, expandOp.src(), composedReassociation);
    return success();
  }
};

template <typename ExpandOpTy, typename CollapseOpTy>
struct ComposeExpandOfCollapseOp : public OpRewritePattern<ExpandOpTy> {
  using OpRewritePattern<ExpandOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(ExpandOpTy expandOp,
                                PatternRewriter &rewriter) const override {
    auto collapseOp = expandOp.src().template getDefiningOp<CollapseOpTy>();
    if (!collapseOp)
      return failure();

    ShapedType srcType = collapseOp.getSrcType();
    ShapedType resultType = expandOp.getResultType();

    if (hasNonIdentityLayout(expandOp.src().getType()) ||
        hasNonIdentityLayout(collapseOp.src().getType()) ||
        hasNonIdentityLayout(collapseOp.result().getType()))
      return failure();

    int64_t srcRank = srcType.getRank();
    int64_t resultRank = resultType.getRank();
    if (srcType == resultType)
      return failure();

    auto srcReassociation = collapseOp.getReassociationIndices();
    auto resultReassociation = expandOp.getReassociationIndices();
    if (srcRank > resultRank) {
      auto composedReassociation = findCollapsingReassociation(
          srcReassociation, resultReassociation, srcType.getShape(),
          resultType.getShape());
      if (!composedReassociation.hasValue())
        return failure();

      rewriter.replaceOpWithNewOp<CollapseOpTy>(
          expandOp, resultType, collapseOp.src(), *composedReassociation);
      return success();
    }
    auto composedReassociation =
        findCollapsingReassociation(resultReassociation, srcReassociation,
                                    resultType.getShape(), srcType.getShape());
    if (!composedReassociation.hasValue())
      return failure();

    rewriter.replaceOpWithNewOp<ExpandOpTy>(
        expandOp, resultType, collapseOp.src(), *composedReassociation);
    return success();
  }

private:
  // Attempts to find a way to collapse `srcShape` to `resultShape` by
  // collapsing subshapes defined by the reassociation indices.
  Optional<SmallVector<ReassociationIndices>> findCollapsingReassociation(
      ArrayRef<ReassociationIndices> srcReassociation,
      ArrayRef<ReassociationIndices> resultReassociation,
      ArrayRef<int64_t> srcShape, ArrayRef<int64_t> resultShape) const {
    SmallVector<ReassociationIndices, 4> composedReassociation;

    if (srcReassociation.empty())
      return {getReassociationIndicesForCollapse(srcShape, resultShape)};

    for (auto item : llvm::zip(srcReassociation, resultReassociation)) {
      auto &srcIndices = std::get<0>(item);
      auto &resultIndices = std::get<1>(item);
      auto srcSubShape = srcShape.slice(srcIndices.front(), srcIndices.size());
      auto resultSubShape =
          resultShape.slice(resultIndices.front(), resultIndices.size());

      if (srcSubShape.size() == resultSubShape.size()) {
        if (srcSubShape == resultSubShape)
          composedReassociation.push_back(srcIndices);
        else
          return llvm::None;
      }

      // Find reassociation to collapse `srcSubShape` into `resultSubShape`.
      auto subShapeReassociation =
          getReassociationIndicesForCollapse(srcSubShape, resultSubShape);
      if (!subShapeReassociation.hasValue())
        return llvm::None;

      // Remap the subshape indices back to the original srcShape.
      for (auto &subshape_indices : *subShapeReassociation) {
        ReassociationIndices shape_indices;
        for (int64_t index : subshape_indices)
          shape_indices.push_back(srcIndices.front() + index);
        composedReassociation.push_back(shape_indices);
      }
    }
    return {std::move(composedReassociation)};
  }
};

} // namespace mlir

#endif // MLIR_DIALECT_UTILS_RESHAPEOPSUTILS_H
