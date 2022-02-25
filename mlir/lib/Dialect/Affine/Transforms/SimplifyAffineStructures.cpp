//===- SimplifyAffineStructures.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to simplify affine structures in operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Utils.h"

#define DEBUG_TYPE "simplify-affine-structure"

using namespace mlir;

namespace {

/// Simplifies affine maps and sets appearing in the operations of the Function.
/// This part is mainly to test the simplifyAffineExpr method. In addition,
/// all memrefs with non-trivial layout maps are converted to ones with trivial
/// identity layout ones.
struct SimplifyAffineStructures
    : public SimplifyAffineStructuresBase<SimplifyAffineStructures> {
  void runOnFunction() override;

  /// Utility to simplify an affine attribute and update its entry in the parent
  /// operation if necessary.
  template <typename AttributeT>
  void simplifyAndUpdateAttribute(Operation *op, Identifier name,
                                  AttributeT attr) {
    auto &simplified = simplifiedAttributes[attr];
    if (simplified == attr)
      return;

    // This is a newly encountered attribute.
    if (!simplified) {
      // Try to simplify the value of the attribute.
      auto value = attr.getValue();
      auto simplifiedValue = simplify(value);
      if (simplifiedValue == value) {
        simplified = attr;
        return;
      }
      simplified = AttributeT::get(simplifiedValue);
    }

    // Simplification was successful, so update the attribute.
    op->setAttr(name, simplified);
  }

  IntegerSet simplify(IntegerSet set) { return simplifyIntegerSet(set); }

  /// Performs basic affine map simplifications.
  AffineMap simplify(AffineMap map) {
    MutableAffineMap mMap(map);
    mMap.simplify();
    return mMap.getAffineMap();
  }

  DenseMap<Attribute, Attribute> simplifiedAttributes;
};

} // end anonymous namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createSimplifyAffineStructuresPass() {
  return std::make_unique<SimplifyAffineStructures>();
}

void SimplifyAffineStructures::runOnFunction() {
  auto func = getFunction();
  simplifiedAttributes.clear();
  RewritePatternSet patterns(func.getContext());
  AffineApplyOp::getCanonicalizationPatterns(patterns, func.getContext());
  AffineForOp::getCanonicalizationPatterns(patterns, func.getContext());
  AffineIfOp::getCanonicalizationPatterns(patterns, func.getContext());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  // The simplification of affine attributes will likely simplify the op. Try to
  // fold/apply canonicalization patterns when we have affine dialect ops.
  SmallVector<Operation *> opsToSimplify;
  func.walk([&](Operation *op) {
    for (auto attr : op->getAttrs()) {
      if (auto mapAttr = attr.second.dyn_cast<AffineMapAttr>())
        simplifyAndUpdateAttribute(op, attr.first, mapAttr);
      else if (auto setAttr = attr.second.dyn_cast<IntegerSetAttr>())
        simplifyAndUpdateAttribute(op, attr.first, setAttr);
    }

    if (isa<AffineForOp, AffineIfOp, AffineApplyOp>(op))
      opsToSimplify.push_back(op);
  });
  (void)applyOpPatternsAndFold(opsToSimplify, frozenPatterns, /*strict=*/true);
}
