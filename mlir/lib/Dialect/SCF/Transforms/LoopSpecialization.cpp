//===- LoopSpecialization.cpp - scf.parallel/SCR.for specialization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Specializes parallel loops and for loops for easier unrolling and
// vectorization.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using scf::ForOp;
using scf::ParallelOp;

/// Rewrite a parallel loop with bounds defined by an affine.min with a constant
/// into 2 loops after checking if the bounds are equal to that constant. This
/// is beneficial if the loop will almost always have the constant bound and
/// that version can be fully unrolled and vectorized.
static void specializeParallelLoopForUnrolling(ParallelOp op) {
  SmallVector<int64_t, 2> constantIndices;
  constantIndices.reserve(op.upperBound().size());
  for (auto bound : op.upperBound()) {
    auto minOp = bound.getDefiningOp<AffineMinOp>();
    if (!minOp)
      return;
    int64_t minConstant = std::numeric_limits<int64_t>::max();
    for (AffineExpr expr : minOp.map().getResults()) {
      if (auto constantIndex = expr.dyn_cast<AffineConstantExpr>())
        minConstant = std::min(minConstant, constantIndex.getValue());
    }
    if (minConstant == std::numeric_limits<int64_t>::max())
      return;
    constantIndices.push_back(minConstant);
  }

  OpBuilder b(op);
  BlockAndValueMapping map;
  Value cond;
  for (auto bound : llvm::zip(op.upperBound(), constantIndices)) {
    Value constant = b.create<ConstantIndexOp>(op.getLoc(), std::get<1>(bound));
    Value cmp = b.create<CmpIOp>(op.getLoc(), CmpIPredicate::eq,
                                 std::get<0>(bound), constant);
    cond = cond ? b.create<AndOp>(op.getLoc(), cond, cmp) : cmp;
    map.map(std::get<0>(bound), constant);
  }
  auto ifOp = b.create<scf::IfOp>(op.getLoc(), cond, /*withElseRegion=*/true);
  ifOp.getThenBodyBuilder().clone(*op.getOperation(), map);
  ifOp.getElseBodyBuilder().clone(*op.getOperation());
  op.erase();
}

/// Rewrite a for loop with bounds defined by an affine.min with a constant into
/// 2 loops after checking if the bounds are equal to that constant. This is
/// beneficial if the loop will almost always have the constant bound and that
/// version can be fully unrolled and vectorized.
static void specializeForLoopForUnrolling(ForOp op) {
  auto bound = op.upperBound();
  auto minOp = bound.getDefiningOp<AffineMinOp>();
  if (!minOp)
    return;
  int64_t minConstant = std::numeric_limits<int64_t>::max();
  for (AffineExpr expr : minOp.map().getResults()) {
    if (auto constantIndex = expr.dyn_cast<AffineConstantExpr>())
      minConstant = std::min(minConstant, constantIndex.getValue());
  }
  if (minConstant == std::numeric_limits<int64_t>::max())
    return;

  OpBuilder b(op);
  BlockAndValueMapping map;
  Value constant = b.create<ConstantIndexOp>(op.getLoc(), minConstant);
  Value cond =
      b.create<CmpIOp>(op.getLoc(), CmpIPredicate::eq, bound, constant);
  map.map(bound, constant);
  auto ifOp = b.create<scf::IfOp>(op.getLoc(), cond, /*withElseRegion=*/true);
  ifOp.getThenBodyBuilder().clone(*op.getOperation(), map);
  ifOp.getElseBodyBuilder().clone(*op.getOperation());
  op.erase();
}

/// Rewrite a for loop with bounds/step that potentially do not divide evenly
/// into a for loop where the step divides the iteration space evenly, followed
/// by an scf.if for the last (partial) iteration (if any).
///
/// This function rewrites the given scf.for loop in-place and creates a new
/// scf.if operation for the last iteration. It replaces all uses of the
/// unpeeled loop with the results of the newly generated scf.if.
///
/// The newly generated scf.if operation is returned via `ifOp`. The boundary
/// at which the loop is split (new upper bound) is returned via `splitBound`.
/// The return value indicates whether the loop was rewritten or not.
static LogicalResult peelForLoop(RewriterBase &b, ForOp forOp, scf::IfOp &ifOp,
                                 Value &splitBound) {
  RewriterBase::InsertionGuard guard(b);
  auto lbInt = getConstantIntValue(forOp.lowerBound());
  auto ubInt = getConstantIntValue(forOp.upperBound());
  auto stepInt = getConstantIntValue(forOp.step());

  // No specialization necessary if step already divides upper bound evenly.
  if (lbInt && ubInt && stepInt && (*ubInt - *lbInt) % *stepInt == 0)
    return failure();
  // No specialization necessary if step size is 1.
  if (stepInt == static_cast<int64_t>(1))
    return failure();

  auto loc = forOp.getLoc();
  AffineExpr sym0, sym1, sym2;
  bindSymbols(b.getContext(), sym0, sym1, sym2);
  // New upper bound: %ub - (%ub - %lb) mod %step
  auto modMap = AffineMap::get(0, 3, {sym1 - ((sym1 - sym0) % sym2)});
  b.setInsertionPoint(forOp);
  splitBound = b.createOrFold<AffineApplyOp>(
      loc, modMap,
      ValueRange{forOp.lowerBound(), forOp.upperBound(), forOp.step()});

  // Set new upper loop bound.
  Value previousUb = forOp.upperBound();
  b.updateRootInPlace(forOp,
                      [&]() { forOp.upperBoundMutable().assign(splitBound); });
  b.setInsertionPointAfter(forOp);

  // Do we need one more iteration?
  Value hasMoreIter =
      b.create<CmpIOp>(loc, CmpIPredicate::slt, splitBound, previousUb);

  // Create IfOp for last iteration.
  auto resultTypes = forOp.getResultTypes();
  ifOp = b.create<scf::IfOp>(loc, resultTypes, hasMoreIter,
                             /*withElseRegion=*/!resultTypes.empty());
  forOp.replaceAllUsesWith(ifOp->getResults());

  // Build then case.
  BlockAndValueMapping bvm;
  bvm.map(forOp.region().getArgument(0), splitBound);
  for (auto it : llvm::zip(forOp.getRegionIterArgs(), forOp->getResults())) {
    bvm.map(std::get<0>(it), std::get<1>(it));
  }
  b.cloneRegionBefore(forOp.region(), ifOp.thenRegion(),
                      ifOp.thenRegion().begin(), bvm);
  // Build else case.
  if (!resultTypes.empty())
    ifOp.getElseBodyBuilder(b.getListener())
        .create<scf::YieldOp>(loc, forOp->getResults());

  return success();
}

static void unpackOptionalValues(ArrayRef<Optional<Value>> source,
                                 SmallVector<Value> &target) {
  target = llvm::to_vector<4>(llvm::map_range(source, [](Optional<Value> val) {
    return val.hasValue() ? *val : Value();
  }));
}

/// Bound an identifier `pos` in a given FlatAffineValueConstraints with
/// constraints drawn from an affine map. Before adding the constraint, the
/// dimensions/symbols of the affine map are aligned with `constraints`.
/// `operands` are the SSA Value operands used with the affine map.
/// Note: This function adds a new symbol column to the `constraints` for each
/// dimension/symbol that exists in the affine map but not in `constraints`.
static LogicalResult alignAndAddBound(FlatAffineValueConstraints &constraints,
                                      FlatAffineConstraints::BoundType type,
                                      unsigned pos, AffineMap map,
                                      ValueRange operands) {
  SmallVector<Value> dims, syms, newSyms;
  unpackOptionalValues(constraints.getMaybeDimValues(), dims);
  unpackOptionalValues(constraints.getMaybeSymbolValues(), syms);

  AffineMap alignedMap =
      alignAffineMapWithValues(map, operands, dims, syms, &newSyms);
  for (unsigned i = syms.size(); i < newSyms.size(); ++i)
    constraints.addSymbolId(constraints.getNumSymbolIds(), newSyms[i]);
  return constraints.addBound(type, pos, alignedMap);
}

/// This function tries to canonicalize min/max operations by proving that their
/// value is bounded by the same lower and upper bound. In that case, the
/// operation can be folded away.
///
/// Bounds are computed by FlatAffineValueConstraints. Invariants required for
/// finding/proving bounds should be supplied via `constraints`.
///
/// 1. Add dimensions for `op` and `opBound` (lower or upper bound of `op`).
/// 2. Compute an upper bound of `op` (in case of `isMin`) or a lower bound (in
///    case of `!isMin`) and bind it to `opBound`. SSA values that are used in
///    `op` but are not part of `constraints`, are added as extra symbols.
/// 3. For each result of `op`: Add result as a dimension `r_i`. Prove that:
///    * If `isMin`: r_i >= opBound
///    * If `isMax`: r_i <= opBound
///    If this is the case, ub(op) == lb(op).
/// 4. Replace `op` with `opBound`.
///
/// In summary, the following constraints are added throughout this function.
/// Note: `invar` are dimensions added by the caller to express the invariants.
/// (Showing only the case where `isMin`.)
///
///  invar |    op | opBound | r_i | extra syms... | const |           eq/ineq
///  ------+-------+---------+-----+---------------+-------+-------------------
///   (various eq./ineq. constraining `invar`, added by the caller)
///    ... |     0 |       0 |   0 |             0 |   ... |               ...
///  ------+-------+---------+-----+---------------+-------+-------------------
///   (various ineq. constraining `op` in terms of `op` operands (`invar` and
///    extra `op` operands "extra syms" that are not in `invar`)).
///    ... |    -1 |       0 |   0 |           ... |   ... |              >= 0
///  ------+-------+---------+-----+---------------+-------+-------------------
///   (set `opBound` to `op` upper bound in terms of `invar` and "extra syms")
///    ... |     0 |      -1 |   0 |           ... |   ... |               = 0
///  ------+-------+---------+-----+---------------+-------+-------------------
///   (for each `op` map result r_i: set r_i to corresponding map result,
///    prove that r_i >= minOpUb via contradiction)
///    ... |     0 |       0 |  -1 |           ... |   ... |               = 0
///      0 |     0 |       1 |  -1 |             0 |    -1 |              >= 0
///
static LogicalResult
canonicalizeMinMaxOp(RewriterBase &rewriter, Operation *op, AffineMap map,
                     ValueRange operands, bool isMin,
                     FlatAffineValueConstraints constraints) {
  RewriterBase::InsertionGuard guard(rewriter);
  unsigned numResults = map.getNumResults();

  // Add a few extra dimensions.
  unsigned dimOp = constraints.addDimId();      // `op`
  unsigned dimOpBound = constraints.addDimId(); // `op` lower/upper bound
  unsigned resultDimStart = constraints.getNumDimIds();
  for (unsigned i = 0; i < numResults; ++i)
    constraints.addDimId();

  // Add an inequality for each result expr_i of map:
  // isMin: op <= expr_i, !isMin: op >= expr_i
  auto boundType =
      isMin ? FlatAffineConstraints::UB : FlatAffineConstraints::LB;
  if (failed(alignAndAddBound(constraints, boundType, dimOp, map, operands)))
    return failure();

  // Try to compute a lower/upper bound for op, expressed in terms of the other
  // `dims` and extra symbols.
  SmallVector<AffineMap> opLb(1), opUb(1);
  constraints.getSliceBounds(dimOp, 1, rewriter.getContext(), &opLb, &opUb);
  AffineMap boundMap = isMin ? opUb[0] : opLb[0];
  // TODO: `getSliceBounds` may return multiple bounds at the moment. This is
  // a TODO of `getSliceBounds` and not handled here.
  if (!boundMap || boundMap.getNumResults() != 1)
    return failure(); // No or multiple bounds found.

  // Add an equality: Set dimOpBound to computed bound.
  // Add back dimension for op. (Was removed by `getSliceBounds`.)
  AffineMap alignedBoundMap = boundMap.shiftDims(/*shift=*/1, /*offset=*/dimOp);
  if (failed(constraints.addBound(FlatAffineConstraints::EQ, dimOpBound,
                                  alignedBoundMap)))
    return failure();

  // If the constraint system is empty, there is an inconsistency. (E.g., this
  // can happen if loop lb > ub.)
  if (constraints.isEmpty())
    return failure();

  // In the case of `isMin` (`!isMin` is inversed):
  // Prove that each result of `map` has a lower bound that is equal to (or
  // greater than) the upper bound of `op` (`dimOpBound`). In that case, `op`
  // can be replaced with the bound. I.e., prove that for each result
  // expr_i (represented by dimension r_i):
  //
  // r_i >= opBound
  //
  // To prove this inequality, add its negation to the constraint set and prove
  // that the constraint set is empty.
  for (unsigned i = resultDimStart; i < resultDimStart + numResults; ++i) {
    FlatAffineValueConstraints newConstr(constraints);

    // Add an equality: r_i = expr_i
    // Note: These equalities could have been added earlier and used to express
    // minOp <= expr_i. However, then we run the risk that `getSliceBounds`
    // computes minOpUb in terms of r_i dims, which is not desired.
    if (failed(alignAndAddBound(newConstr, FlatAffineConstraints::EQ, i,
                                map.getSubMap({i - resultDimStart}), operands)))
      return failure();

    // If `isMin`:  Add inequality: r_i < opBound
    //              equiv.: opBound - r_i - 1 >= 0
    // If `!isMin`: Add inequality: r_i > opBound
    //              equiv.: -opBound + r_i - 1 >= 0
    SmallVector<int64_t> ineq(newConstr.getNumCols(), 0);
    ineq[dimOpBound] = isMin ? 1 : -1;
    ineq[i] = isMin ? -1 : 1;
    ineq[newConstr.getNumCols() - 1] = -1;
    newConstr.addInequality(ineq);
    if (!newConstr.isEmpty())
      return failure();
  }

  // Lower and upper bound of `op` are equal. Replace `minOp` with its bound.
  AffineMap newMap = alignedBoundMap;
  SmallVector<Value> newOperands;
  unpackOptionalValues(constraints.getMaybeDimAndSymbolValues(), newOperands);
  mlir::canonicalizeMapAndOperands(&newMap, &newOperands);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<AffineApplyOp>(op, newMap, newOperands);
  return success();
}

/// Try to simplify a min/max operation `op` after loop peeling. This function
/// can simplify min/max operations such as (ub is the previous upper bound of
/// the unpeeled loop):
/// ```
/// #map = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
/// %r = affine.min #affine.min #map(%iv)[%step, %ub]
/// ```
/// and rewrites them into (in the case the peeled loop):
/// ```
/// %r = %step
/// ```
/// min/max operations inside the generated scf.if operation are rewritten in
/// a similar way.
///
/// This function builds up a set of constraints, capable of proving that:
/// * Inside the peeled loop: min(step, ub - iv) == step
/// * Inside the scf.if operation: min(step, ub - iv) == ub - iv
///
/// Returns `success` if the given operation was replaced by a new operation;
/// `failure` otherwise.
///
/// Note: `ub` is the previous upper bound of the loop (before peeling).
/// `insideLoop` must be true for min/max ops inside the loop and false for
/// affine.min ops inside the scf.for op. For an explanation of the other
/// parameters, see comment of `canonicalizeMinMaxOpInLoop`.
static LogicalResult rewritePeeledMinMaxOp(RewriterBase &rewriter,
                                           Operation *op, AffineMap map,
                                           ValueRange operands, bool isMin,
                                           Value iv, Value ub, Value step,
                                           bool insideLoop) {
  FlatAffineValueConstraints constraints;
  constraints.addDimId(0, iv);
  constraints.addDimId(1, ub);
  constraints.addDimId(2, step);
  if (auto constUb = getConstantIntValue(ub))
    constraints.addBound(FlatAffineConstraints::EQ, 1, *constUb);
  if (auto constStep = getConstantIntValue(step))
    constraints.addBound(FlatAffineConstraints::EQ, 2, *constStep);

  // Add loop peeling invariant. This is the main piece of knowledge that
  // enables AffineMinOp simplification.
  if (insideLoop) {
    // ub - iv >= step (equiv.: -iv + ub - step + 0 >= 0)
    // Intuitively: Inside the peeled loop, every iteration is a "full"
    // iteration, i.e., step divides the iteration space `ub - lb` evenly.
    constraints.addInequality({-1, 1, -1, 0});
  } else {
    // ub - iv < step (equiv.: iv + -ub + step - 1 >= 0)
    // Intuitively: `iv` is the split bound here, i.e., the iteration variable
    // value of the very last iteration (in the unpeeled loop). At that point,
    // there are less than `step` elements remaining. (Otherwise, the peeled
    // loop would run for at least one more iteration.)
    constraints.addInequality({1, -1, 1, -1});
  }

  return canonicalizeMinMaxOp(rewriter, op, map, operands, isMin, constraints);
}

template <typename OpTy, bool IsMin>
static void
rewriteAffineOpAfterPeeling(RewriterBase &rewriter, ForOp forOp, scf::IfOp ifOp,
                            Value iv, Value splitBound, Value ub, Value step) {
  forOp.walk([&](OpTy affineOp) {
    (void)rewritePeeledMinMaxOp(rewriter, affineOp, affineOp.getAffineMap(),
                                affineOp.operands(), IsMin, iv, ub, step,
                                /*insideLoop=*/true);
  });
  ifOp.walk([&](OpTy affineOp) {
    (void)rewritePeeledMinMaxOp(rewriter, affineOp, affineOp.getAffineMap(),
                                affineOp.operands(), IsMin, splitBound, ub,
                                step, /*insideLoop=*/false);
  });
}

LogicalResult mlir::scf::peelAndCanonicalizeForLoop(RewriterBase &rewriter,
                                                    ForOp forOp,
                                                    scf::IfOp &ifOp) {
  Value ub = forOp.upperBound();
  Value splitBound;
  if (failed(peelForLoop(rewriter, forOp, ifOp, splitBound)))
    return failure();

  // Rewrite affine.min and affine.max ops.
  Value iv = forOp.getInductionVar(), step = forOp.step();
  rewriteAffineOpAfterPeeling<AffineMinOp, /*IsMin=*/true>(
      rewriter, forOp, ifOp, iv, splitBound, ub, step);
  rewriteAffineOpAfterPeeling<AffineMaxOp, /*IsMin=*/false>(
      rewriter, forOp, ifOp, iv, splitBound, ub, step);

  return success();
}

/// Canonicalize min/max operations in the context of for loops with a known
/// range. Call `canonicalizeMinMaxOp` and add the following constraints to
/// the constraint system (along with the missing dimensions):
///
/// * iv >= lb
/// * iv < lb + step * ((ub - lb - 1) floorDiv step) + 1
///
/// Note: Due to limitations of FlatAffineConstraints, only constant step sizes
/// are currently supported.
LogicalResult
mlir::scf::canonicalizeMinMaxOpInLoop(RewriterBase &rewriter, Operation *op,
                                      AffineMap map, ValueRange operands,
                                      bool isMin, LoopMatcherFn loopMatcher) {
  FlatAffineValueConstraints constraints;
  DenseSet<Value> allIvs;

  // Find all iteration variables among `minOp`'s operands add constrain them.
  for (Value operand : operands) {
    // Skip duplicate ivs.
    if (llvm::find(allIvs, operand) != allIvs.end())
      continue;

    // If `operand` is an iteration variable: Find corresponding loop
    // bounds and step.
    Value iv = operand;
    Value lb, ub, step;
    if (failed(loopMatcher(operand, lb, ub, step)))
      continue;
    allIvs.insert(iv);

    // FlatAffineConstraints does not support semi-affine expressions.
    // Therefore, only constant step values are supported.
    auto stepInt = getConstantIntValue(step);
    if (!stepInt)
      continue;

    unsigned dimIv = constraints.addDimId(iv);
    unsigned dimLb = constraints.addDimId(lb);
    unsigned dimUb = constraints.addDimId(ub);

    // If loop lower/upper bounds are constant: Add EQ constraint.
    Optional<int64_t> lbInt = getConstantIntValue(lb);
    Optional<int64_t> ubInt = getConstantIntValue(ub);
    if (lbInt)
      constraints.addBound(FlatAffineConstraints::EQ, dimLb, *lbInt);
    if (ubInt)
      constraints.addBound(FlatAffineConstraints::EQ, dimUb, *ubInt);

    // iv >= lb (equiv.: iv - lb >= 0)
    SmallVector<int64_t> ineqLb(constraints.getNumCols(), 0);
    ineqLb[dimIv] = 1;
    ineqLb[dimLb] = -1;
    constraints.addInequality(ineqLb);

    // iv < lb + step * ((ub - lb - 1) floorDiv step) + 1
    AffineExpr exprLb = lbInt ? rewriter.getAffineConstantExpr(*lbInt)
                              : rewriter.getAffineDimExpr(dimLb);
    AffineExpr exprUb = ubInt ? rewriter.getAffineConstantExpr(*ubInt)
                              : rewriter.getAffineDimExpr(dimUb);
    AffineExpr ivUb =
        exprLb + 1 + (*stepInt * ((exprUb - exprLb - 1).floorDiv(*stepInt)));
    auto map = AffineMap::get(
        /*dimCount=*/constraints.getNumDimIds(),
        /*symbolCount=*/constraints.getNumSymbolIds(), /*result=*/ivUb);

    if (failed(constraints.addBound(FlatAffineConstraints::UB, dimIv, map)))
      return failure();
  }

  return canonicalizeMinMaxOp(rewriter, op, map, operands, isMin, constraints);
}

static constexpr char kPeeledLoopLabel[] = "__peeled_loop__";
static constexpr char kPartialIterationLabel[] = "__partial_iteration__";

namespace {
struct ForLoopPeelingPattern : public OpRewritePattern<ForOp> {
  ForLoopPeelingPattern(MLIRContext *ctx, bool skipPartial)
      : OpRewritePattern<ForOp>(ctx), skipPartial(skipPartial) {}

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Do not peel already peeled loops.
    if (forOp->hasAttr(kPeeledLoopLabel))
      return failure();
    if (skipPartial) {
      // No peeling of loops inside the partial iteration (scf.if) of another
      // peeled loop.
      Operation *op = forOp.getOperation();
      while ((op = op->getParentOfType<scf::IfOp>())) {
        if (op->hasAttr(kPartialIterationLabel))
          return failure();
      }
    }
    // Apply loop peeling.
    scf::IfOp ifOp;
    if (failed(peelAndCanonicalizeForLoop(rewriter, forOp, ifOp)))
      return failure();
    // Apply label, so that the same loop is not rewritten a second time.
    rewriter.updateRootInPlace(forOp, [&]() {
      forOp->setAttr(kPeeledLoopLabel, rewriter.getUnitAttr());
    });
    ifOp->setAttr(kPartialIterationLabel, rewriter.getUnitAttr());
    return success();
  }

  /// If set to true, loops inside partial iterations of another peeled loop
  /// are not peeled. This reduces the size of the generated code. Partial
  /// iterations are not usually performance critical.
  /// Note: Takes into account the entire chain of parent operations, not just
  /// the direct parent.
  bool skipPartial;
};

/// Canonicalize AffineMinOp/AffineMaxOp operations in the context of scf.for
/// and scf.parallel loops with a known range.
template <typename OpTy, bool IsMin>
struct AffineOpSCFCanonicalizationPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto loopMatcher = [](Value iv, Value &lb, Value &ub, Value &step) {
      if (scf::ForOp forOp = scf::getForInductionVarOwner(iv)) {
        lb = forOp.lowerBound();
        ub = forOp.upperBound();
        step = forOp.step();
        return success();
      }
      if (scf::ParallelOp parOp = scf::getParallelForInductionVarOwner(iv)) {
        for (unsigned idx = 0; idx < parOp.getNumLoops(); ++idx) {
          if (parOp.getInductionVars()[idx] == iv) {
            lb = parOp.lowerBound()[idx];
            ub = parOp.upperBound()[idx];
            step = parOp.step()[idx];
            return success();
          }
        }
        return failure();
      }
      return failure();
    };

    return scf::canonicalizeMinMaxOpInLoop(rewriter, op, op.getAffineMap(),
                                           op.operands(), IsMin, loopMatcher);
  }
};
} // namespace

namespace {
struct ParallelLoopSpecialization
    : public SCFParallelLoopSpecializationBase<ParallelLoopSpecialization> {
  void runOnFunction() override {
    getFunction().walk(
        [](ParallelOp op) { specializeParallelLoopForUnrolling(op); });
  }
};

struct ForLoopSpecialization
    : public SCFForLoopSpecializationBase<ForLoopSpecialization> {
  void runOnFunction() override {
    getFunction().walk([](ForOp op) { specializeForLoopForUnrolling(op); });
  }
};

struct ForLoopPeeling : public SCFForLoopPeelingBase<ForLoopPeeling> {
  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ForLoopPeelingPattern>(ctx, skipPartial);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

    // Drop the markers.
    funcOp.walk([](Operation *op) {
      op->removeAttr(kPeeledLoopLabel);
      op->removeAttr(kPartialIterationLabel);
    });
  }
};

struct SCFAffineOpCanonicalization
    : public SCFAffineOpCanonicalizationBase<SCFAffineOpCanonicalization> {
  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    scf::populateSCFLoopBodyCanonicalizationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createSCFAffineOpCanonicalizationPass() {
  return std::make_unique<SCFAffineOpCanonicalization>();
}

std::unique_ptr<Pass> mlir::createParallelLoopSpecializationPass() {
  return std::make_unique<ParallelLoopSpecialization>();
}

std::unique_ptr<Pass> mlir::createForLoopSpecializationPass() {
  return std::make_unique<ForLoopSpecialization>();
}

std::unique_ptr<Pass> mlir::createForLoopPeelingPass() {
  return std::make_unique<ForLoopPeeling>();
}

void mlir::scf::populateSCFLoopBodyCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns
      .insert<AffineOpSCFCanonicalizationPattern<AffineMinOp, /*IsMin=*/true>,
              AffineOpSCFCanonicalizationPattern<AffineMaxOp, /*IsMin=*/false>>(
          ctx);
}
