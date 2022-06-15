//===- AffineExpr.cpp - MLIR Affine Expr Classes --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/IR/AffineExpr.h"
#include "AffineExprDetail.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::detail;

MLIRContext *AffineExpr::getContext() const { return expr->context; }

AffineExprKind AffineExpr::getKind() const { return expr->kind; }

/// Walk all of the AffineExprs in this subgraph in postorder.
void AffineExpr::walk(std::function<void(AffineExpr)> callback) const {
  struct AffineExprWalker : public AffineExprVisitor<AffineExprWalker> {
    std::function<void(AffineExpr)> callback;

    AffineExprWalker(std::function<void(AffineExpr)> callback)
        : callback(std::move(callback)) {}

    void visitAffineBinaryOpExpr(AffineBinaryOpExpr expr) { callback(expr); }
    void visitConstantExpr(AffineConstantExpr expr) { callback(expr); }
    void visitDimExpr(AffineDimExpr expr) { callback(expr); }
    void visitSymbolExpr(AffineSymbolExpr expr) { callback(expr); }
  };

  AffineExprWalker(std::move(callback)).walkPostOrder(*this);
}

// Dispatch affine expression construction based on kind.
AffineExpr mlir::getAffineBinaryOpExpr(AffineExprKind kind, AffineExpr lhs,
                                       AffineExpr rhs) {
  if (kind == AffineExprKind::Add)
    return lhs + rhs;
  if (kind == AffineExprKind::Mul)
    return lhs * rhs;
  if (kind == AffineExprKind::FloorDiv)
    return lhs.floorDiv(rhs);
  if (kind == AffineExprKind::CeilDiv)
    return lhs.ceilDiv(rhs);
  if (kind == AffineExprKind::Mod)
    return lhs % rhs;

  llvm_unreachable("unknown binary operation on affine expressions");
}

/// This method substitutes any uses of dimensions and symbols (e.g.
/// dim#0 with dimReplacements[0]) and returns the modified expression tree.
AffineExpr
AffineExpr::replaceDimsAndSymbols(ArrayRef<AffineExpr> dimReplacements,
                                  ArrayRef<AffineExpr> symReplacements) const {
  switch (getKind()) {
  case AffineExprKind::Constant:
    return *this;
  case AffineExprKind::DimId: {
    unsigned dimId = cast<AffineDimExpr>().getPosition();
    if (dimId >= dimReplacements.size())
      return *this;
    return dimReplacements[dimId];
  }
  case AffineExprKind::SymbolId: {
    unsigned symId = cast<AffineSymbolExpr>().getPosition();
    if (symId >= symReplacements.size())
      return *this;
    return symReplacements[symId];
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod:
    auto binOp = cast<AffineBinaryOpExpr>();
    auto lhs = binOp.getLHS(), rhs = binOp.getRHS();
    auto newLHS = lhs.replaceDimsAndSymbols(dimReplacements, symReplacements);
    auto newRHS = rhs.replaceDimsAndSymbols(dimReplacements, symReplacements);
    if (newLHS == lhs && newRHS == rhs)
      return *this;
    return getAffineBinaryOpExpr(getKind(), newLHS, newRHS);
  }
  llvm_unreachable("Unknown AffineExpr");
}

AffineExpr AffineExpr::replaceDims(ArrayRef<AffineExpr> dimReplacements) const {
  return replaceDimsAndSymbols(dimReplacements, {});
}

AffineExpr
AffineExpr::replaceSymbols(ArrayRef<AffineExpr> symReplacements) const {
  return replaceDimsAndSymbols({}, symReplacements);
}

/// Replace dims[offset ... numDims)
/// by dims[offset + shift ... shift + numDims).
AffineExpr AffineExpr::shiftDims(unsigned numDims, unsigned shift,
                                 unsigned offset) const {
  SmallVector<AffineExpr, 4> dims;
  for (unsigned idx = 0; idx < offset; ++idx)
    dims.push_back(getAffineDimExpr(idx, getContext()));
  for (unsigned idx = offset; idx < numDims; ++idx)
    dims.push_back(getAffineDimExpr(idx + shift, getContext()));
  return replaceDimsAndSymbols(dims, {});
}

/// Replace symbols[offset ... numSymbols)
/// by symbols[offset + shift ... shift + numSymbols).
AffineExpr AffineExpr::shiftSymbols(unsigned numSymbols, unsigned shift,
                                    unsigned offset) const {
  SmallVector<AffineExpr, 4> symbols;
  for (unsigned idx = 0; idx < offset; ++idx)
    symbols.push_back(getAffineSymbolExpr(idx, getContext()));
  for (unsigned idx = offset; idx < numSymbols; ++idx)
    symbols.push_back(getAffineSymbolExpr(idx + shift, getContext()));
  return replaceDimsAndSymbols({}, symbols);
}

/// Sparse replace method. Return the modified expression tree.
AffineExpr
AffineExpr::replace(const DenseMap<AffineExpr, AffineExpr> &map) const {
  auto it = map.find(*this);
  if (it != map.end())
    return it->second;
  switch (getKind()) {
  default:
    return *this;
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod:
    auto binOp = cast<AffineBinaryOpExpr>();
    auto lhs = binOp.getLHS(), rhs = binOp.getRHS();
    auto newLHS = lhs.replace(map);
    auto newRHS = rhs.replace(map);
    if (newLHS == lhs && newRHS == rhs)
      return *this;
    return getAffineBinaryOpExpr(getKind(), newLHS, newRHS);
  }
  llvm_unreachable("Unknown AffineExpr");
}

/// Sparse replace method. Return the modified expression tree.
AffineExpr AffineExpr::replace(AffineExpr expr, AffineExpr replacement) const {
  DenseMap<AffineExpr, AffineExpr> map;
  map.insert(std::make_pair(expr, replacement));
  return replace(map);
}
/// Returns true if this expression is made out of only symbols and
/// constants (no dimensional identifiers).
bool AffineExpr::isSymbolicOrConstant() const {
  switch (getKind()) {
  case AffineExprKind::Constant:
    return true;
  case AffineExprKind::DimId:
    return false;
  case AffineExprKind::SymbolId:
    return true;

  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    auto expr = this->cast<AffineBinaryOpExpr>();
    return expr.getLHS().isSymbolicOrConstant() &&
           expr.getRHS().isSymbolicOrConstant();
  }
  }
  llvm_unreachable("Unknown AffineExpr");
}

/// Returns true if this is a pure affine expression, i.e., multiplication,
/// floordiv, ceildiv, and mod is only allowed w.r.t constants.
bool AffineExpr::isPureAffine() const {
  switch (getKind()) {
  case AffineExprKind::SymbolId:
  case AffineExprKind::DimId:
  case AffineExprKind::Constant:
    return true;
  case AffineExprKind::Add: {
    auto op = cast<AffineBinaryOpExpr>();
    return op.getLHS().isPureAffine() && op.getRHS().isPureAffine();
  }

  case AffineExprKind::Mul: {
    // TODO: Canonicalize the constants in binary operators to the RHS when
    // possible, allowing this to merge into the next case.
    auto op = cast<AffineBinaryOpExpr>();
    return op.getLHS().isPureAffine() && op.getRHS().isPureAffine() &&
           (op.getLHS().template isa<AffineConstantExpr>() ||
            op.getRHS().template isa<AffineConstantExpr>());
  }
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    auto op = cast<AffineBinaryOpExpr>();
    return op.getLHS().isPureAffine() &&
           op.getRHS().template isa<AffineConstantExpr>();
  }
  }
  llvm_unreachable("Unknown AffineExpr");
}

// Returns the greatest known integral divisor of this affine expression.
int64_t AffineExpr::getLargestKnownDivisor() const {
  AffineBinaryOpExpr binExpr(nullptr);
  switch (getKind()) {
  case AffineExprKind::CeilDiv:
    LLVM_FALLTHROUGH;
  case AffineExprKind::DimId:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::SymbolId:
    return 1;
  case AffineExprKind::Constant:
    return std::abs(this->cast<AffineConstantExpr>().getValue());
  case AffineExprKind::Mul: {
    binExpr = this->cast<AffineBinaryOpExpr>();
    return binExpr.getLHS().getLargestKnownDivisor() *
           binExpr.getRHS().getLargestKnownDivisor();
  }
  case AffineExprKind::Add:
    LLVM_FALLTHROUGH;
  case AffineExprKind::Mod: {
    binExpr = cast<AffineBinaryOpExpr>();
    return llvm::GreatestCommonDivisor64(
        binExpr.getLHS().getLargestKnownDivisor(),
        binExpr.getRHS().getLargestKnownDivisor());
  }
  }
  llvm_unreachable("Unknown AffineExpr");
}

bool AffineExpr::isMultipleOf(int64_t factor) const {
  AffineBinaryOpExpr binExpr(nullptr);
  uint64_t l, u;
  switch (getKind()) {
  case AffineExprKind::SymbolId:
    LLVM_FALLTHROUGH;
  case AffineExprKind::DimId:
    return factor * factor == 1;
  case AffineExprKind::Constant:
    return cast<AffineConstantExpr>().getValue() % factor == 0;
  case AffineExprKind::Mul: {
    binExpr = cast<AffineBinaryOpExpr>();
    // It's probably not worth optimizing this further (to not traverse the
    // whole sub-tree under - it that would require a version of isMultipleOf
    // that on a 'false' return also returns the largest known divisor).
    return (l = binExpr.getLHS().getLargestKnownDivisor()) % factor == 0 ||
           (u = binExpr.getRHS().getLargestKnownDivisor()) % factor == 0 ||
           (l * u) % factor == 0;
  }
  case AffineExprKind::Add:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    binExpr = cast<AffineBinaryOpExpr>();
    return llvm::GreatestCommonDivisor64(
               binExpr.getLHS().getLargestKnownDivisor(),
               binExpr.getRHS().getLargestKnownDivisor()) %
               factor ==
           0;
  }
  }
  llvm_unreachable("Unknown AffineExpr");
}

bool AffineExpr::isFunctionOfDim(unsigned position) const {
  if (getKind() == AffineExprKind::DimId) {
    return *this == mlir::getAffineDimExpr(position, getContext());
  }
  if (auto expr = this->dyn_cast<AffineBinaryOpExpr>()) {
    return expr.getLHS().isFunctionOfDim(position) ||
           expr.getRHS().isFunctionOfDim(position);
  }
  return false;
}

bool AffineExpr::isFunctionOfSymbol(unsigned position) const {
  if (getKind() == AffineExprKind::SymbolId) {
    return *this == mlir::getAffineSymbolExpr(position, getContext());
  }
  if (auto expr = this->dyn_cast<AffineBinaryOpExpr>()) {
    return expr.getLHS().isFunctionOfSymbol(position) ||
           expr.getRHS().isFunctionOfSymbol(position);
  }
  return false;
}

AffineBinaryOpExpr::AffineBinaryOpExpr(AffineExpr::ImplType *ptr)
    : AffineExpr(ptr) {}
AffineExpr AffineBinaryOpExpr::getLHS() const {
  return static_cast<ImplType *>(expr)->lhs;
}
AffineExpr AffineBinaryOpExpr::getRHS() const {
  return static_cast<ImplType *>(expr)->rhs;
}

AffineDimExpr::AffineDimExpr(AffineExpr::ImplType *ptr) : AffineExpr(ptr) {}
unsigned AffineDimExpr::getPosition() const {
  return static_cast<ImplType *>(expr)->position;
}

/// Returns true if the expression is divisible by the given symbol with
/// position `symbolPos`. The argument `opKind` specifies here what kind of
/// division or mod operation called this division. It helps in implementing the
/// commutative property of the floordiv and ceildiv operations. If the argument
///`exprKind` is floordiv and `expr` is also a binary expression of a floordiv
/// operation, then the commutative property can be used otherwise, the floordiv
/// operation is not divisible. The same argument holds for ceildiv operation.
static bool isDivisibleBySymbol(AffineExpr expr, unsigned symbolPos,
                                AffineExprKind opKind) {
  // The argument `opKind` can either be Modulo, Floordiv or Ceildiv only.
  assert((opKind == AffineExprKind::Mod || opKind == AffineExprKind::FloorDiv ||
          opKind == AffineExprKind::CeilDiv) &&
         "unexpected opKind");
  switch (expr.getKind()) {
  case AffineExprKind::Constant:
    return expr.cast<AffineConstantExpr>().getValue() == 0;
  case AffineExprKind::DimId:
    return false;
  case AffineExprKind::SymbolId:
    return (expr.cast<AffineSymbolExpr>().getPosition() == symbolPos);
  // Checks divisibility by the given symbol for both operands.
  case AffineExprKind::Add: {
    AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
    return isDivisibleBySymbol(binaryExpr.getLHS(), symbolPos, opKind) &&
           isDivisibleBySymbol(binaryExpr.getRHS(), symbolPos, opKind);
  }
  // Checks divisibility by the given symbol for both operands. Consider the
  // expression `(((s1*s0) floordiv w) mod ((s1 * s2) floordiv p)) floordiv s1`,
  // this is a division by s1 and both the operands of modulo are divisible by
  // s1 but it is not divisible by s1 always. The third argument is
  // `AffineExprKind::Mod` for this reason.
  case AffineExprKind::Mod: {
    AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
    return isDivisibleBySymbol(binaryExpr.getLHS(), symbolPos,
                               AffineExprKind::Mod) &&
           isDivisibleBySymbol(binaryExpr.getRHS(), symbolPos,
                               AffineExprKind::Mod);
  }
  // Checks if any of the operand divisible by the given symbol.
  case AffineExprKind::Mul: {
    AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
    return isDivisibleBySymbol(binaryExpr.getLHS(), symbolPos, opKind) ||
           isDivisibleBySymbol(binaryExpr.getRHS(), symbolPos, opKind);
  }
  // Floordiv and ceildiv are divisible by the given symbol when the first
  // operand is divisible, and the affine expression kind of the argument expr
  // is same as the argument `opKind`. This can be inferred from commutative
  // property of floordiv and ceildiv operations and are as follow:
  // (exp1 floordiv exp2) floordiv exp3 = (exp1 floordiv exp3) floordiv exp2
  // (exp1 ceildiv exp2) ceildiv exp3 = (exp1 ceildiv exp3) ceildiv expr2
  // It will fail if operations are not same. For example:
  // (exps1 ceildiv exp2) floordiv exp3 can not be simplified.
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
    AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
    if (opKind != expr.getKind())
      return false;
    return isDivisibleBySymbol(binaryExpr.getLHS(), symbolPos, expr.getKind());
  }
  }
  llvm_unreachable("Unknown AffineExpr");
}

/// Divides the given expression by the given symbol at position `symbolPos`. It
/// considers the divisibility condition is checked before calling itself. A
/// null expression is returned whenever the divisibility condition fails.
static AffineExpr symbolicDivide(AffineExpr expr, unsigned symbolPos,
                                 AffineExprKind opKind) {
  // THe argument `opKind` can either be Modulo, Floordiv or Ceildiv only.
  assert((opKind == AffineExprKind::Mod || opKind == AffineExprKind::FloorDiv ||
          opKind == AffineExprKind::CeilDiv) &&
         "unexpected opKind");
  switch (expr.getKind()) {
  case AffineExprKind::Constant:
    if (expr.cast<AffineConstantExpr>().getValue() != 0)
      return nullptr;
    return getAffineConstantExpr(0, expr.getContext());
  case AffineExprKind::DimId:
    return nullptr;
  case AffineExprKind::SymbolId:
    return getAffineConstantExpr(1, expr.getContext());
  // Dividing both operands by the given symbol.
  case AffineExprKind::Add: {
    AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
    return getAffineBinaryOpExpr(
        expr.getKind(), symbolicDivide(binaryExpr.getLHS(), symbolPos, opKind),
        symbolicDivide(binaryExpr.getRHS(), symbolPos, opKind));
  }
  // Dividing both operands by the given symbol.
  case AffineExprKind::Mod: {
    AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
    return getAffineBinaryOpExpr(
        expr.getKind(),
        symbolicDivide(binaryExpr.getLHS(), symbolPos, expr.getKind()),
        symbolicDivide(binaryExpr.getRHS(), symbolPos, expr.getKind()));
  }
  // Dividing any of the operand by the given symbol.
  case AffineExprKind::Mul: {
    AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
    if (!isDivisibleBySymbol(binaryExpr.getLHS(), symbolPos, opKind))
      return binaryExpr.getLHS() *
             symbolicDivide(binaryExpr.getRHS(), symbolPos, opKind);
    return symbolicDivide(binaryExpr.getLHS(), symbolPos, opKind) *
           binaryExpr.getRHS();
  }
  // Dividing first operand only by the given symbol.
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
    AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
    return getAffineBinaryOpExpr(
        expr.getKind(),
        symbolicDivide(binaryExpr.getLHS(), symbolPos, expr.getKind()),
        binaryExpr.getRHS());
  }
  }
  llvm_unreachable("Unknown AffineExpr");
}

/// Simplify a semi-affine expression by handling modulo, floordiv, or ceildiv
/// operations when the second operand simplifies to a symbol and the first
/// operand is divisible by that symbol. It can be applied to any semi-affine
/// expression. Returned expression can either be a semi-affine or pure affine
/// expression.
static AffineExpr simplifySemiAffine(AffineExpr expr) {
  switch (expr.getKind()) {
  case AffineExprKind::Constant:
  case AffineExprKind::DimId:
  case AffineExprKind::SymbolId:
    return expr;
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
    return getAffineBinaryOpExpr(expr.getKind(),
                                 simplifySemiAffine(binaryExpr.getLHS()),
                                 simplifySemiAffine(binaryExpr.getRHS()));
  }
  // Check if the simplification of the second operand is a symbol, and the
  // first operand is divisible by it. If the operation is a modulo, a constant
  // zero expression is returned. In the case of floordiv and ceildiv, the
  // symbol from the simplification of the second operand divides the first
  // operand. Otherwise, simplification is not possible.
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
    AffineExpr sLHS = simplifySemiAffine(binaryExpr.getLHS());
    AffineExpr sRHS = simplifySemiAffine(binaryExpr.getRHS());
    AffineSymbolExpr symbolExpr =
        simplifySemiAffine(binaryExpr.getRHS()).dyn_cast<AffineSymbolExpr>();
    if (!symbolExpr)
      return getAffineBinaryOpExpr(expr.getKind(), sLHS, sRHS);
    unsigned symbolPos = symbolExpr.getPosition();
    if (!isDivisibleBySymbol(binaryExpr.getLHS(), symbolPos, expr.getKind()))
      return getAffineBinaryOpExpr(expr.getKind(), sLHS, sRHS);
    if (expr.getKind() == AffineExprKind::Mod)
      return getAffineConstantExpr(0, expr.getContext());
    return symbolicDivide(sLHS, symbolPos, expr.getKind());
  }
  }
  llvm_unreachable("Unknown AffineExpr");
}

static AffineExpr getAffineDimOrSymbol(AffineExprKind kind, unsigned position,
                                       MLIRContext *context) {
  auto assignCtx = [context](AffineDimExprStorage *storage) {
    storage->context = context;
  };

  StorageUniquer &uniquer = context->getAffineUniquer();
  return uniquer.get<AffineDimExprStorage>(
      assignCtx, static_cast<unsigned>(kind), position);
}

AffineExpr mlir::getAffineDimExpr(unsigned position, MLIRContext *context) {
  return getAffineDimOrSymbol(AffineExprKind::DimId, position, context);
}

AffineSymbolExpr::AffineSymbolExpr(AffineExpr::ImplType *ptr)
    : AffineExpr(ptr) {}
unsigned AffineSymbolExpr::getPosition() const {
  return static_cast<ImplType *>(expr)->position;
}

AffineExpr mlir::getAffineSymbolExpr(unsigned position, MLIRContext *context) {
  return getAffineDimOrSymbol(AffineExprKind::SymbolId, position, context);
  ;
}

AffineConstantExpr::AffineConstantExpr(AffineExpr::ImplType *ptr)
    : AffineExpr(ptr) {}
int64_t AffineConstantExpr::getValue() const {
  return static_cast<ImplType *>(expr)->constant;
}

bool AffineExpr::operator==(int64_t v) const {
  return *this == getAffineConstantExpr(v, getContext());
}

AffineExpr mlir::getAffineConstantExpr(int64_t constant, MLIRContext *context) {
  auto assignCtx = [context](AffineConstantExprStorage *storage) {
    storage->context = context;
  };

  StorageUniquer &uniquer = context->getAffineUniquer();
  return uniquer.get<AffineConstantExprStorage>(assignCtx, constant);
}

/// Simplify add expression. Return nullptr if it can't be simplified.
static AffineExpr simplifyAdd(AffineExpr lhs, AffineExpr rhs) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExpr>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExpr>();
  // Fold if both LHS, RHS are a constant.
  if (lhsConst && rhsConst)
    return getAffineConstantExpr(lhsConst.getValue() + rhsConst.getValue(),
                                 lhs.getContext());

  // Canonicalize so that only the RHS is a constant. (4 + d0 becomes d0 + 4).
  // If only one of them is a symbolic expressions, make it the RHS.
  if (lhs.isa<AffineConstantExpr>() ||
      (lhs.isSymbolicOrConstant() && !rhs.isSymbolicOrConstant())) {
    return rhs + lhs;
  }

  // At this point, if there was a constant, it would be on the right.

  // Addition with a zero is a noop, return the other input.
  if (rhsConst) {
    if (rhsConst.getValue() == 0)
      return lhs;
  }
  // Fold successive additions like (d0 + 2) + 3 into d0 + 5.
  auto lBin = lhs.dyn_cast<AffineBinaryOpExpr>();
  if (lBin && rhsConst && lBin.getKind() == AffineExprKind::Add) {
    if (auto lrhs = lBin.getRHS().dyn_cast<AffineConstantExpr>())
      return lBin.getLHS() + (lrhs.getValue() + rhsConst.getValue());
  }

  // Detect "c1 * expr + c_2 * expr" as "(c1 + c2) * expr".
  // c1 is rRhsConst, c2 is rLhsConst; firstExpr, secondExpr are their
  // respective multiplicands.
  Optional<int64_t> rLhsConst, rRhsConst;
  AffineExpr firstExpr, secondExpr;
  AffineConstantExpr rLhsConstExpr;
  auto lBinOpExpr = lhs.dyn_cast<AffineBinaryOpExpr>();
  if (lBinOpExpr && lBinOpExpr.getKind() == AffineExprKind::Mul &&
      (rLhsConstExpr = lBinOpExpr.getRHS().dyn_cast<AffineConstantExpr>())) {
    rLhsConst = rLhsConstExpr.getValue();
    firstExpr = lBinOpExpr.getLHS();
  } else {
    rLhsConst = 1;
    firstExpr = lhs;
  }

  auto rBinOpExpr = rhs.dyn_cast<AffineBinaryOpExpr>();
  AffineConstantExpr rRhsConstExpr;
  if (rBinOpExpr && rBinOpExpr.getKind() == AffineExprKind::Mul &&
      (rRhsConstExpr = rBinOpExpr.getRHS().dyn_cast<AffineConstantExpr>())) {
    rRhsConst = rRhsConstExpr.getValue();
    secondExpr = rBinOpExpr.getLHS();
  } else {
    rRhsConst = 1;
    secondExpr = rhs;
  }

  if (rLhsConst && rRhsConst && firstExpr == secondExpr)
    return getAffineBinaryOpExpr(
        AffineExprKind::Mul, firstExpr,
        getAffineConstantExpr(rLhsConst.getValue() + rRhsConst.getValue(),
                              lhs.getContext()));

  // When doing successive additions, bring constant to the right: turn (d0 + 2)
  // + d1 into (d0 + d1) + 2.
  if (lBin && lBin.getKind() == AffineExprKind::Add) {
    if (auto lrhs = lBin.getRHS().dyn_cast<AffineConstantExpr>()) {
      return lBin.getLHS() + rhs + lrhs;
    }
  }

  // Detect and transform "expr - q * (expr floordiv q)" to "expr mod q", where
  // q may be a constant or symbolic expression. This leads to a much more
  // efficient form when 'c' is a power of two, and in general a more compact
  // and readable form.

  // Process '(expr floordiv c) * (-c)'.
  if (!rBinOpExpr)
    return nullptr;

  auto lrhs = rBinOpExpr.getLHS();
  auto rrhs = rBinOpExpr.getRHS();

  AffineExpr llrhs, rlrhs;

  // Check if lrhsBinOpExpr is of the form (expr floordiv q) * q, where q is a
  // symbolic expression.
  auto lrhsBinOpExpr = lrhs.dyn_cast<AffineBinaryOpExpr>();
  // Check rrhsConstOpExpr = -1.
  auto rrhsConstOpExpr = rrhs.dyn_cast<AffineConstantExpr>();
  if (rrhsConstOpExpr && rrhsConstOpExpr.getValue() == -1 && lrhsBinOpExpr &&
      lrhsBinOpExpr.getKind() == AffineExprKind::Mul) {
    // Check llrhs = expr floordiv q.
    llrhs = lrhsBinOpExpr.getLHS();
    // Check rlrhs = q.
    rlrhs = lrhsBinOpExpr.getRHS();
    auto llrhsBinOpExpr = llrhs.dyn_cast<AffineBinaryOpExpr>();
    if (!llrhsBinOpExpr || llrhsBinOpExpr.getKind() != AffineExprKind::FloorDiv)
      return nullptr;
    if (llrhsBinOpExpr.getRHS() == rlrhs && lhs == llrhsBinOpExpr.getLHS())
      return lhs % rlrhs;
  }

  // Process lrhs, which is 'expr floordiv c'.
  AffineBinaryOpExpr lrBinOpExpr = lrhs.dyn_cast<AffineBinaryOpExpr>();
  if (!lrBinOpExpr || lrBinOpExpr.getKind() != AffineExprKind::FloorDiv)
    return nullptr;

  llrhs = lrBinOpExpr.getLHS();
  rlrhs = lrBinOpExpr.getRHS();

  if (lhs == llrhs && rlrhs == -rrhs) {
    return lhs % rlrhs;
  }
  return nullptr;
}

AffineExpr AffineExpr::operator+(int64_t v) const {
  return *this + getAffineConstantExpr(v, getContext());
}
AffineExpr AffineExpr::operator+(AffineExpr other) const {
  if (auto simplified = simplifyAdd(*this, other))
    return simplified;

  StorageUniquer &uniquer = getContext()->getAffineUniquer();
  return uniquer.get<AffineBinaryOpExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(AffineExprKind::Add), *this, other);
}

/// Simplify a multiply expression. Return nullptr if it can't be simplified.
static AffineExpr simplifyMul(AffineExpr lhs, AffineExpr rhs) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExpr>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExpr>();

  if (lhsConst && rhsConst)
    return getAffineConstantExpr(lhsConst.getValue() * rhsConst.getValue(),
                                 lhs.getContext());

  assert(lhs.isSymbolicOrConstant() || rhs.isSymbolicOrConstant());

  // Canonicalize the mul expression so that the constant/symbolic term is the
  // RHS. If both the lhs and rhs are symbolic, swap them if the lhs is a
  // constant. (Note that a constant is trivially symbolic).
  if (!rhs.isSymbolicOrConstant() || lhs.isa<AffineConstantExpr>()) {
    // At least one of them has to be symbolic.
    return rhs * lhs;
  }

  // At this point, if there was a constant, it would be on the right.

  // Multiplication with a one is a noop, return the other input.
  if (rhsConst) {
    if (rhsConst.getValue() == 1)
      return lhs;
    // Multiplication with zero.
    if (rhsConst.getValue() == 0)
      return rhsConst;
  }

  // Fold successive multiplications: eg: (d0 * 2) * 3 into d0 * 6.
  auto lBin = lhs.dyn_cast<AffineBinaryOpExpr>();
  if (lBin && rhsConst && lBin.getKind() == AffineExprKind::Mul) {
    if (auto lrhs = lBin.getRHS().dyn_cast<AffineConstantExpr>())
      return lBin.getLHS() * (lrhs.getValue() * rhsConst.getValue());
  }

  // When doing successive multiplication, bring constant to the right: turn (d0
  // * 2) * d1 into (d0 * d1) * 2.
  if (lBin && lBin.getKind() == AffineExprKind::Mul) {
    if (auto lrhs = lBin.getRHS().dyn_cast<AffineConstantExpr>()) {
      return (lBin.getLHS() * rhs) * lrhs;
    }
  }

  return nullptr;
}

AffineExpr AffineExpr::operator*(int64_t v) const {
  return *this * getAffineConstantExpr(v, getContext());
}
AffineExpr AffineExpr::operator*(AffineExpr other) const {
  if (auto simplified = simplifyMul(*this, other))
    return simplified;

  StorageUniquer &uniquer = getContext()->getAffineUniquer();
  return uniquer.get<AffineBinaryOpExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(AffineExprKind::Mul), *this, other);
}

// Unary minus, delegate to operator*.
AffineExpr AffineExpr::operator-() const {
  return *this * getAffineConstantExpr(-1, getContext());
}

// Delegate to operator+.
AffineExpr AffineExpr::operator-(int64_t v) const { return *this + (-v); }
AffineExpr AffineExpr::operator-(AffineExpr other) const {
  return *this + (-other);
}

static AffineExpr simplifyFloorDiv(AffineExpr lhs, AffineExpr rhs) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExpr>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExpr>();

  // mlir floordiv by zero or negative numbers is undefined and preserved as is.
  if (!rhsConst || rhsConst.getValue() < 1)
    return nullptr;

  if (lhsConst)
    return getAffineConstantExpr(
        floorDiv(lhsConst.getValue(), rhsConst.getValue()), lhs.getContext());

  // Fold floordiv of a multiply with a constant that is a multiple of the
  // divisor. Eg: (i * 128) floordiv 64 = i * 2.
  if (rhsConst == 1)
    return lhs;

  // Simplify (expr * const) floordiv divConst when expr is known to be a
  // multiple of divConst.
  auto lBin = lhs.dyn_cast<AffineBinaryOpExpr>();
  if (lBin && lBin.getKind() == AffineExprKind::Mul) {
    if (auto lrhs = lBin.getRHS().dyn_cast<AffineConstantExpr>()) {
      // rhsConst is known to be a positive constant.
      if (lrhs.getValue() % rhsConst.getValue() == 0)
        return lBin.getLHS() * (lrhs.getValue() / rhsConst.getValue());
    }
  }

  // Simplify (expr1 + expr2) floordiv divConst when either expr1 or expr2 is
  // known to be a multiple of divConst.
  if (lBin && lBin.getKind() == AffineExprKind::Add) {
    int64_t llhsDiv = lBin.getLHS().getLargestKnownDivisor();
    int64_t lrhsDiv = lBin.getRHS().getLargestKnownDivisor();
    // rhsConst is known to be a positive constant.
    if (llhsDiv % rhsConst.getValue() == 0 ||
        lrhsDiv % rhsConst.getValue() == 0)
      return lBin.getLHS().floorDiv(rhsConst.getValue()) +
             lBin.getRHS().floorDiv(rhsConst.getValue());
  }

  return nullptr;
}

AffineExpr AffineExpr::floorDiv(uint64_t v) const {
  return floorDiv(getAffineConstantExpr(v, getContext()));
}
AffineExpr AffineExpr::floorDiv(AffineExpr other) const {
  if (auto simplified = simplifyFloorDiv(*this, other))
    return simplified;

  StorageUniquer &uniquer = getContext()->getAffineUniquer();
  return uniquer.get<AffineBinaryOpExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(AffineExprKind::FloorDiv), *this,
      other);
}

static AffineExpr simplifyCeilDiv(AffineExpr lhs, AffineExpr rhs) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExpr>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExpr>();

  if (!rhsConst || rhsConst.getValue() < 1)
    return nullptr;

  if (lhsConst)
    return getAffineConstantExpr(
        ceilDiv(lhsConst.getValue(), rhsConst.getValue()), lhs.getContext());

  // Fold ceildiv of a multiply with a constant that is a multiple of the
  // divisor. Eg: (i * 128) ceildiv 64 = i * 2.
  if (rhsConst.getValue() == 1)
    return lhs;

  // Simplify (expr * const) ceildiv divConst when const is known to be a
  // multiple of divConst.
  auto lBin = lhs.dyn_cast<AffineBinaryOpExpr>();
  if (lBin && lBin.getKind() == AffineExprKind::Mul) {
    if (auto lrhs = lBin.getRHS().dyn_cast<AffineConstantExpr>()) {
      // rhsConst is known to be a positive constant.
      if (lrhs.getValue() % rhsConst.getValue() == 0)
        return lBin.getLHS() * (lrhs.getValue() / rhsConst.getValue());
    }
  }

  return nullptr;
}

AffineExpr AffineExpr::ceilDiv(uint64_t v) const {
  return ceilDiv(getAffineConstantExpr(v, getContext()));
}
AffineExpr AffineExpr::ceilDiv(AffineExpr other) const {
  if (auto simplified = simplifyCeilDiv(*this, other))
    return simplified;

  StorageUniquer &uniquer = getContext()->getAffineUniquer();
  return uniquer.get<AffineBinaryOpExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(AffineExprKind::CeilDiv), *this,
      other);
}

static AffineExpr simplifyMod(AffineExpr lhs, AffineExpr rhs) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExpr>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExpr>();

  // mod w.r.t zero or negative numbers is undefined and preserved as is.
  if (!rhsConst || rhsConst.getValue() < 1)
    return nullptr;

  if (lhsConst)
    return getAffineConstantExpr(mod(lhsConst.getValue(), rhsConst.getValue()),
                                 lhs.getContext());

  // Fold modulo of an expression that is known to be a multiple of a constant
  // to zero if that constant is a multiple of the modulo factor. Eg: (i * 128)
  // mod 64 is folded to 0, and less trivially, (i*(j*4*(k*32))) mod 128 = 0.
  if (lhs.getLargestKnownDivisor() % rhsConst.getValue() == 0)
    return getAffineConstantExpr(0, lhs.getContext());

  // Simplify (expr1 + expr2) mod divConst when either expr1 or expr2 is
  // known to be a multiple of divConst.
  auto lBin = lhs.dyn_cast<AffineBinaryOpExpr>();
  if (lBin && lBin.getKind() == AffineExprKind::Add) {
    int64_t llhsDiv = lBin.getLHS().getLargestKnownDivisor();
    int64_t lrhsDiv = lBin.getRHS().getLargestKnownDivisor();
    // rhsConst is known to be a positive constant.
    if (llhsDiv % rhsConst.getValue() == 0)
      return lBin.getRHS() % rhsConst.getValue();
    if (lrhsDiv % rhsConst.getValue() == 0)
      return lBin.getLHS() % rhsConst.getValue();
  }

  // Simplify (e % a) % b to e % b when b evenly divides a
  if (lBin && lBin.getKind() == AffineExprKind::Mod) {
    auto intermediate = lBin.getRHS().dyn_cast<AffineConstantExpr>();
    if (intermediate && intermediate.getValue() >= 1 &&
        mod(intermediate.getValue(), rhsConst.getValue()) == 0) {
      return lBin.getLHS() % rhsConst.getValue();
    }
  }

  return nullptr;
}

AffineExpr AffineExpr::operator%(uint64_t v) const {
  return *this % getAffineConstantExpr(v, getContext());
}
AffineExpr AffineExpr::operator%(AffineExpr other) const {
  if (auto simplified = simplifyMod(*this, other))
    return simplified;

  StorageUniquer &uniquer = getContext()->getAffineUniquer();
  return uniquer.get<AffineBinaryOpExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(AffineExprKind::Mod), *this, other);
}

AffineExpr AffineExpr::compose(AffineMap map) const {
  SmallVector<AffineExpr, 8> dimReplacements(map.getResults().begin(),
                                             map.getResults().end());
  return replaceDimsAndSymbols(dimReplacements, {});
}
raw_ostream &mlir::operator<<(raw_ostream &os, AffineExpr expr) {
  expr.print(os);
  return os;
}

/// Constructs an affine expression from a flat ArrayRef. If there are local
/// identifiers (neither dimensional nor symbolic) that appear in the sum of
/// products expression, `localExprs` is expected to have the AffineExpr
/// for it, and is substituted into. The ArrayRef `flatExprs` is expected to be
/// in the format [dims, symbols, locals, constant term].
AffineExpr mlir::getAffineExprFromFlatForm(ArrayRef<int64_t> flatExprs,
                                           unsigned numDims,
                                           unsigned numSymbols,
                                           ArrayRef<AffineExpr> localExprs,
                                           MLIRContext *context) {
  // Assert expected numLocals = flatExprs.size() - numDims - numSymbols - 1.
  assert(flatExprs.size() - numDims - numSymbols - 1 == localExprs.size() &&
         "unexpected number of local expressions");

  auto expr = getAffineConstantExpr(0, context);
  // Dimensions and symbols.
  for (unsigned j = 0; j < numDims + numSymbols; j++) {
    if (flatExprs[j] == 0)
      continue;
    auto id = j < numDims ? getAffineDimExpr(j, context)
                          : getAffineSymbolExpr(j - numDims, context);
    expr = expr + id * flatExprs[j];
  }

  // Local identifiers.
  for (unsigned j = numDims + numSymbols, e = flatExprs.size() - 1; j < e;
       j++) {
    if (flatExprs[j] == 0)
      continue;
    auto term = localExprs[j - numDims - numSymbols] * flatExprs[j];
    expr = expr + term;
  }

  // Constant term.
  int64_t constTerm = flatExprs[flatExprs.size() - 1];
  if (constTerm != 0)
    expr = expr + constTerm;
  return expr;
}

/// Constructs a semi-affine expression from a flat ArrayRef. If there are
/// local identifiers (neither dimensional nor symbolic) that appear in the sum
/// of products expression, `localExprs` is expected to have the AffineExprs for
/// it, and is substituted into. The ArrayRef `flatExprs` is expected to be in
/// the format [dims, symbols, locals, constant term]. The semi-affine
/// expression is constructed in the sorted order of dimension and symbol
/// position numbers. Note:  local expressions/ids are used for mod, div as well
/// as symbolic RHS terms for terms that are not pure affine.
static AffineExpr getSemiAffineExprFromFlatForm(ArrayRef<int64_t> flatExprs,
                                                unsigned numDims,
                                                unsigned numSymbols,
                                                ArrayRef<AffineExpr> localExprs,
                                                MLIRContext *context) {
  assert(!flatExprs.empty() && "flatExprs cannot be empty");

  // Assert expected numLocals = flatExprs.size() - numDims - numSymbols - 1.
  assert(flatExprs.size() - numDims - numSymbols - 1 == localExprs.size() &&
         "unexpected number of local expressions");

  AffineExpr expr = getAffineConstantExpr(0, context);

  // We design indices as a pair which help us present the semi-affine map as
  // sum of product where terms are sorted based on dimension or symbol
  // position: <keyA, keyB> for expressions of the form dimension * symbol,
  // where keyA is the position number of the dimension and keyB is the
  // position number of the symbol. For dimensional expressions we set the index
  // as (position number of the dimension, -1), as we want dimensional
  // expressions to appear before symbolic and product of dimensional and
  // symbolic expressions having the dimension with the same position number.
  // For symbolic expression set the index as (position number of the symbol,
  // maximum of last dimension and symbol position) number. For example, we want
  // the expression we are constructing to look something like: d0 + d0 * s0 +
  // s0 + d1*s1 + s1.

  // Stores the affine expression corresponding to a given index.
  DenseMap<std::pair<unsigned, signed>, AffineExpr> indexToExprMap;
  // Stores the constant coefficient value corresponding to a given
  // dimension, symbol or a non-pure affine expression stored in `localExprs`.
  DenseMap<std::pair<unsigned, signed>, int64_t> coefficients;
  // Stores the indices as defined above, and later sorted to produce
  // the semi-affine expression in the desired form.
  SmallVector<std::pair<unsigned, signed>, 8> indices;

  // Example: expression = d0 + d0 * s0 + 2 * s0.
  // indices = [{0,-1}, {0, 0}, {0, 1}]
  // coefficients = [{{0, -1}, 1}, {{0, 0}, 1}, {{0, 1}, 2}]
  // indexToExprMap = [{{0, -1}, d0}, {{0, 0}, d0 * s0}, {{0, 1}, s0}]

  // Adds entries to `indexToExprMap`, `coefficients` and `indices`.
  auto addEntry = [&](std::pair<unsigned, signed> index, int64_t coefficient,
                      AffineExpr expr) {
    assert(std::find(indices.begin(), indices.end(), index) == indices.end() &&
           "Key is already present in indices vector and overwriting will "
           "happen in `indexToExprMap` and `coefficients`!");

    indices.push_back(index);
    coefficients.insert({index, coefficient});
    indexToExprMap.insert({index, expr});
  };

  // Design indices for dimensional or symbolic terms, and store the indices,
  // constant coefficient corresponding to the indices in `coefficients` map,
  // and affine expression corresponding to indices in `indexToExprMap` map.

  for (unsigned j = 0; j < numDims; ++j) {
    if (flatExprs[j] == 0)
      continue;
    // For dimensional expressions we set the index as <position number of the
    // dimension, 0>, as we want dimensional expressions to appear before
    // symbolic ones and products of dimensional and symbolic expressions
    // having the dimension with the same position number.
    std::pair<unsigned, signed> indexEntry(j, -1);
    addEntry(indexEntry, flatExprs[j], getAffineDimExpr(j, context));
  }
  for (unsigned j = numDims; j < numDims + numSymbols; ++j) {
    if (flatExprs[j] == 0)
      continue;
    // For symbolic expression set the index as <position number
    // of the symbol, max(dimCount, symCount)> number,
    // as we want symbolic expressions with the same positional number to
    // appear after dimensional expressions having the same positional number.
    std::pair<unsigned, signed> indexEntry(j - numDims,
                                           std::max(numDims, numSymbols));
    addEntry(indexEntry, flatExprs[j],
             getAffineSymbolExpr(j - numDims, context));
  }

  // Denotes semi-affine product, modulo or division terms, which has been added
  // to the `indexToExpr` map.
  SmallVector<bool, 4> addedToMap(flatExprs.size() - numDims - numSymbols - 1,
                                  false);
  unsigned lhsPos, rhsPos;
  // Construct indices for product terms involving dimension, symbol or constant
  // as lhs/rhs, and store the indices, constant coefficient corresponding to
  // the indices in `coefficients` map, and affine expression corresponding to
  // in indices in `indexToExprMap` map.
  for (const auto &it : llvm::enumerate(localExprs)) {
    AffineExpr expr = it.value();
    if (flatExprs[numDims + numSymbols + it.index()] == 0)
      continue;
    AffineExpr lhs = expr.cast<AffineBinaryOpExpr>().getLHS();
    AffineExpr rhs = expr.cast<AffineBinaryOpExpr>().getRHS();
    if (!((lhs.isa<AffineDimExpr>() || lhs.isa<AffineSymbolExpr>()) &&
          (rhs.isa<AffineDimExpr>() || rhs.isa<AffineSymbolExpr>() ||
           rhs.isa<AffineConstantExpr>()))) {
      continue;
    }
    if (rhs.isa<AffineConstantExpr>()) {
      // For product/modulo/division expressions, when rhs of modulo/division
      // expression is constant, we put 0 in place of keyB, because we want
      // them to appear earlier in the semi-affine expression we are
      // constructing. When rhs is constant, we place 0 in place of keyB.
      if (lhs.isa<AffineDimExpr>()) {
        lhsPos = lhs.cast<AffineDimExpr>().getPosition();
        std::pair<unsigned, signed> indexEntry(lhsPos, -1);
        addEntry(indexEntry, flatExprs[numDims + numSymbols + it.index()],
                 expr);
      } else {
        lhsPos = lhs.cast<AffineSymbolExpr>().getPosition();
        std::pair<unsigned, signed> indexEntry(lhsPos,
                                               std::max(numDims, numSymbols));
        addEntry(indexEntry, flatExprs[numDims + numSymbols + it.index()],
                 expr);
      }
    } else if (lhs.isa<AffineDimExpr>()) {
      // For product/modulo/division expressions having lhs as dimension and rhs
      // as symbol, we order the terms in the semi-affine expression based on
      // the pair: <keyA, keyB> for expressions of the form dimension * symbol,
      // where keyA is the position number of the dimension and keyB is the
      // position number of the symbol.
      lhsPos = lhs.cast<AffineDimExpr>().getPosition();
      rhsPos = rhs.cast<AffineSymbolExpr>().getPosition();
      std::pair<unsigned, signed> indexEntry(lhsPos, rhsPos);
      addEntry(indexEntry, flatExprs[numDims + numSymbols + it.index()], expr);
    } else {
      // For product/modulo/division expressions having both lhs and rhs as
      // symbol, we design indices as a pair: <keyA, keyB> for expressions
      // of the form dimension * symbol, where keyA is the position number of
      // the dimension and keyB is the position number of the symbol.
      lhsPos = lhs.cast<AffineSymbolExpr>().getPosition();
      rhsPos = rhs.cast<AffineSymbolExpr>().getPosition();
      std::pair<unsigned, signed> indexEntry(lhsPos, rhsPos);
      addEntry(indexEntry, flatExprs[numDims + numSymbols + it.index()], expr);
    }
    addedToMap[it.index()] = true;
  }

  // Constructing the simplified semi-affine sum of product/division/mod
  // expression from the flattened form in the desired sorted order of indices
  // of the various individual product/division/mod expressions.
  std::sort(indices.begin(), indices.end());
  for (const std::pair<unsigned, unsigned> index : indices) {
    assert(indexToExprMap.lookup(index) &&
           "cannot find key in `indexToExprMap` map");
    expr = expr + indexToExprMap.lookup(index) * coefficients.lookup(index);
  }

  // Local identifiers.
  for (unsigned j = numDims + numSymbols, e = flatExprs.size() - 1; j < e;
       j++) {
    // If the coefficient of the local expression is 0, continue as we need not
    // add it in out final expression.
    if (flatExprs[j] == 0 || addedToMap[j - numDims - numSymbols])
      continue;
    auto term = localExprs[j - numDims - numSymbols] * flatExprs[j];
    expr = expr + term;
  }

  // Constant term.
  int64_t constTerm = flatExprs.back();
  if (constTerm != 0)
    expr = expr + constTerm;
  return expr;
}

SimpleAffineExprFlattener::SimpleAffineExprFlattener(unsigned numDims,
                                                     unsigned numSymbols)
    : numDims(numDims), numSymbols(numSymbols), numLocals(0) {
  operandExprStack.reserve(8);
}

// In pure affine t = expr * c, we multiply each coefficient of lhs with c.
//
// In case of semi affine multiplication expressions, t = expr * symbolic_expr,
// introduce a local variable p (= expr * symbolic_expr), and the affine
// expression expr * symbolic_expr is added to `localExprs`.
void SimpleAffineExprFlattener::visitMulExpr(AffineBinaryOpExpr expr) {
  assert(operandExprStack.size() >= 2);
  SmallVector<int64_t, 8> rhs = operandExprStack.back();
  operandExprStack.pop_back();
  SmallVector<int64_t, 8> &lhs = operandExprStack.back();

  // Flatten semi-affine multiplication expressions by introducing a local
  // variable in place of the product; the affine expression
  // corresponding to the quantifier is added to `localExprs`.
  if (!expr.getRHS().isa<AffineConstantExpr>()) {
    MLIRContext *context = expr.getContext();
    AffineExpr a = getAffineExprFromFlatForm(lhs, numDims, numSymbols,
                                             localExprs, context);
    AffineExpr b = getAffineExprFromFlatForm(rhs, numDims, numSymbols,
                                             localExprs, context);
    addLocalVariableSemiAffine(a * b, lhs, lhs.size());
    return;
  }

  // Get the RHS constant.
  auto rhsConst = rhs[getConstantIndex()];
  for (unsigned i = 0, e = lhs.size(); i < e; i++) {
    lhs[i] *= rhsConst;
  }
}

void SimpleAffineExprFlattener::visitAddExpr(AffineBinaryOpExpr expr) {
  assert(operandExprStack.size() >= 2);
  const auto &rhs = operandExprStack.back();
  auto &lhs = operandExprStack[operandExprStack.size() - 2];
  assert(lhs.size() == rhs.size());
  // Update the LHS in place.
  for (unsigned i = 0, e = rhs.size(); i < e; i++) {
    lhs[i] += rhs[i];
  }
  // Pop off the RHS.
  operandExprStack.pop_back();
}

//
// t = expr mod c   <=>  t = expr - c*q and c*q <= expr <= c*q + c - 1
//
// A mod expression "expr mod c" is thus flattened by introducing a new local
// variable q (= expr floordiv c), such that expr mod c is replaced with
// 'expr - c * q' and c * q <= expr <= c * q + c - 1 are added to localVarCst.
//
// In case of semi-affine modulo expressions, t = expr mod symbolic_expr,
// introduce a local variable m (= expr mod symbolic_expr), and the affine
// expression expr mod symbolic_expr is added to `localExprs`.
void SimpleAffineExprFlattener::visitModExpr(AffineBinaryOpExpr expr) {
  assert(operandExprStack.size() >= 2);

  SmallVector<int64_t, 8> rhs = operandExprStack.back();
  operandExprStack.pop_back();
  SmallVector<int64_t, 8> &lhs = operandExprStack.back();
  MLIRContext *context = expr.getContext();

  // Flatten semi affine modulo expressions by introducing a local
  // variable in place of the modulo value, and the affine expression
  // corresponding to the quantifier is added to `localExprs`.
  if (!expr.getRHS().isa<AffineConstantExpr>()) {
    AffineExpr dividendExpr = getAffineExprFromFlatForm(
        lhs, numDims, numSymbols, localExprs, context);
    AffineExpr divisorExpr = getAffineExprFromFlatForm(rhs, numDims, numSymbols,
                                                       localExprs, context);
    AffineExpr modExpr = dividendExpr % divisorExpr;
    addLocalVariableSemiAffine(modExpr, lhs, lhs.size());
    return;
  }

  int64_t rhsConst = rhs[getConstantIndex()];
  // TODO: handle modulo by zero case when this issue is fixed
  // at the other places in the IR.
  assert(rhsConst > 0 && "RHS constant has to be positive");

  // Check if the LHS expression is a multiple of modulo factor.
  unsigned i, e;
  for (i = 0, e = lhs.size(); i < e; i++)
    if (lhs[i] % rhsConst != 0)
      break;
  // If yes, modulo expression here simplifies to zero.
  if (i == lhs.size()) {
    std::fill(lhs.begin(), lhs.end(), 0);
    return;
  }

  // Add a local variable for the quotient, i.e., expr % c is replaced by
  // (expr - q * c) where q = expr floordiv c. Do this while canceling out
  // the GCD of expr and c.
  SmallVector<int64_t, 8> floorDividend(lhs);
  uint64_t gcd = rhsConst;
  for (unsigned i = 0, e = lhs.size(); i < e; i++)
    gcd = llvm::GreatestCommonDivisor64(gcd, std::abs(lhs[i]));
  // Simplify the numerator and the denominator.
  if (gcd != 1) {
    for (unsigned i = 0, e = floorDividend.size(); i < e; i++)
      floorDividend[i] = floorDividend[i] / static_cast<int64_t>(gcd);
  }
  int64_t floorDivisor = rhsConst / static_cast<int64_t>(gcd);

  // Construct the AffineExpr form of the floordiv to store in localExprs.

  AffineExpr dividendExpr = getAffineExprFromFlatForm(
      floorDividend, numDims, numSymbols, localExprs, context);
  AffineExpr divisorExpr = getAffineConstantExpr(floorDivisor, context);
  AffineExpr floorDivExpr = dividendExpr.floorDiv(divisorExpr);
  int loc;
  if ((loc = findLocalId(floorDivExpr)) == -1) {
    addLocalFloorDivId(floorDividend, floorDivisor, floorDivExpr);
    // Set result at top of stack to "lhs - rhsConst * q".
    lhs[getLocalVarStartIndex() + numLocals - 1] = -rhsConst;
  } else {
    // Reuse the existing local id.
    lhs[getLocalVarStartIndex() + loc] = -rhsConst;
  }
}

void SimpleAffineExprFlattener::visitCeilDivExpr(AffineBinaryOpExpr expr) {
  visitDivExpr(expr, /*isCeil=*/true);
}
void SimpleAffineExprFlattener::visitFloorDivExpr(AffineBinaryOpExpr expr) {
  visitDivExpr(expr, /*isCeil=*/false);
}

void SimpleAffineExprFlattener::visitDimExpr(AffineDimExpr expr) {
  operandExprStack.emplace_back(SmallVector<int64_t, 32>(getNumCols(), 0));
  auto &eq = operandExprStack.back();
  assert(expr.getPosition() < numDims && "Inconsistent number of dims");
  eq[getDimStartIndex() + expr.getPosition()] = 1;
}

void SimpleAffineExprFlattener::visitSymbolExpr(AffineSymbolExpr expr) {
  operandExprStack.emplace_back(SmallVector<int64_t, 32>(getNumCols(), 0));
  auto &eq = operandExprStack.back();
  assert(expr.getPosition() < numSymbols && "inconsistent number of symbols");
  eq[getSymbolStartIndex() + expr.getPosition()] = 1;
}

void SimpleAffineExprFlattener::visitConstantExpr(AffineConstantExpr expr) {
  operandExprStack.emplace_back(SmallVector<int64_t, 32>(getNumCols(), 0));
  auto &eq = operandExprStack.back();
  eq[getConstantIndex()] = expr.getValue();
}

void SimpleAffineExprFlattener::addLocalVariableSemiAffine(
    AffineExpr expr, SmallVectorImpl<int64_t> &result,
    unsigned long resultSize) {
  assert(result.size() == resultSize &&
         "`result` vector passed is not of correct size");
  int loc;
  if ((loc = findLocalId(expr)) == -1)
    addLocalIdSemiAffine(expr);
  std::fill(result.begin(), result.end(), 0);
  if (loc == -1)
    result[getLocalVarStartIndex() + numLocals - 1] = 1;
  else
    result[getLocalVarStartIndex() + loc] = 1;
}

// t = expr floordiv c   <=> t = q, c * q <= expr <= c * q + c - 1
// A floordiv is thus flattened by introducing a new local variable q, and
// replacing that expression with 'q' while adding the constraints
// c * q <= expr <= c * q + c - 1 to localVarCst (done by
// FlatAffineConstraints::addLocalFloorDiv).
//
// A ceildiv is similarly flattened:
// t = expr ceildiv c   <=> t =  (expr + c - 1) floordiv c
//
// In case of semi affine division expressions, t = expr floordiv symbolic_expr
// or t = expr ceildiv symbolic_expr, introduce a local variable q (= expr
// floordiv/ceildiv symbolic_expr), and the affine floordiv/ceildiv is added to
// `localExprs`.
void SimpleAffineExprFlattener::visitDivExpr(AffineBinaryOpExpr expr,
                                             bool isCeil) {
  assert(operandExprStack.size() >= 2);

  MLIRContext *context = expr.getContext();
  SmallVector<int64_t, 8> rhs = operandExprStack.back();
  operandExprStack.pop_back();
  SmallVector<int64_t, 8> &lhs = operandExprStack.back();

  // Flatten semi affine division expressions by introducing a local
  // variable in place of the quotient, and the affine expression corresponding
  // to the quantifier is added to `localExprs`.
  if (!expr.getRHS().isa<AffineConstantExpr>()) {
    AffineExpr a = getAffineExprFromFlatForm(lhs, numDims, numSymbols,
                                             localExprs, context);
    AffineExpr b = getAffineExprFromFlatForm(rhs, numDims, numSymbols,
                                             localExprs, context);
    AffineExpr divExpr = isCeil ? a.ceilDiv(b) : a.floorDiv(b);
    addLocalVariableSemiAffine(divExpr, lhs, lhs.size());
    return;
  }

  // This is a pure affine expr; the RHS is a positive constant.
  int64_t rhsConst = rhs[getConstantIndex()];
  // TODO: handle division by zero at the same time the issue is
  // fixed at other places.
  assert(rhsConst > 0 && "RHS constant has to be positive");

  // Simplify the floordiv, ceildiv if possible by canceling out the greatest
  // common divisors of the numerator and denominator.
  uint64_t gcd = std::abs(rhsConst);
  for (unsigned i = 0, e = lhs.size(); i < e; i++)
    gcd = llvm::GreatestCommonDivisor64(gcd, std::abs(lhs[i]));
  // Simplify the numerator and the denominator.
  if (gcd != 1) {
    for (unsigned i = 0, e = lhs.size(); i < e; i++)
      lhs[i] = lhs[i] / static_cast<int64_t>(gcd);
  }
  int64_t divisor = rhsConst / static_cast<int64_t>(gcd);
  // If the divisor becomes 1, the updated LHS is the result. (The
  // divisor can't be negative since rhsConst is positive).
  if (divisor == 1)
    return;

  // If the divisor cannot be simplified to one, we will have to retain
  // the ceil/floor expr (simplified up until here). Add an existential
  // quantifier to express its result, i.e., expr1 div expr2 is replaced
  // by a new identifier, q.
  AffineExpr a =
      getAffineExprFromFlatForm(lhs, numDims, numSymbols, localExprs, context);
  AffineExpr b = getAffineConstantExpr(divisor, context);

  int loc;
  AffineExpr divExpr = isCeil ? a.ceilDiv(b) : a.floorDiv(b);
  if ((loc = findLocalId(divExpr)) == -1) {
    if (!isCeil) {
      SmallVector<int64_t, 8> dividend(lhs);
      addLocalFloorDivId(dividend, divisor, divExpr);
    } else {
      // lhs ceildiv c <=>  (lhs + c - 1) floordiv c
      SmallVector<int64_t, 8> dividend(lhs);
      dividend.back() += divisor - 1;
      addLocalFloorDivId(dividend, divisor, divExpr);
    }
  }
  // Set the expression on stack to the local var introduced to capture the
  // result of the division (floor or ceil).
  std::fill(lhs.begin(), lhs.end(), 0);
  if (loc == -1)
    lhs[getLocalVarStartIndex() + numLocals - 1] = 1;
  else
    lhs[getLocalVarStartIndex() + loc] = 1;
}

// Add a local identifier (needed to flatten a mod, floordiv, ceildiv expr).
// The local identifier added is always a floordiv of a pure add/mul affine
// function of other identifiers, coefficients of which are specified in
// dividend and with respect to a positive constant divisor. localExpr is the
// simplified tree expression (AffineExpr) corresponding to the quantifier.
void SimpleAffineExprFlattener::addLocalFloorDivId(ArrayRef<int64_t> dividend,
                                                   int64_t divisor,
                                                   AffineExpr localExpr) {
  assert(divisor > 0 && "positive constant divisor expected");
  for (SmallVector<int64_t, 8> &subExpr : operandExprStack)
    subExpr.insert(subExpr.begin() + getLocalVarStartIndex() + numLocals, 0);
  localExprs.push_back(localExpr);
  numLocals++;
  // dividend and divisor are not used here; an override of this method uses it.
}

void SimpleAffineExprFlattener::addLocalIdSemiAffine(AffineExpr localExpr) {
  for (SmallVector<int64_t, 8> &subExpr : operandExprStack)
    subExpr.insert(subExpr.begin() + getLocalVarStartIndex() + numLocals, 0);
  localExprs.push_back(localExpr);
  ++numLocals;
}

int SimpleAffineExprFlattener::findLocalId(AffineExpr localExpr) {
  SmallVectorImpl<AffineExpr>::iterator it;
  if ((it = llvm::find(localExprs, localExpr)) == localExprs.end())
    return -1;
  return it - localExprs.begin();
}

/// Simplify the affine expression by flattening it and reconstructing it.
AffineExpr mlir::simplifyAffineExpr(AffineExpr expr, unsigned numDims,
                                    unsigned numSymbols) {
  // Simplify semi-affine expressions separately.
  if (!expr.isPureAffine())
    expr = simplifySemiAffine(expr);

  SimpleAffineExprFlattener flattener(numDims, numSymbols);
  flattener.walkPostOrder(expr);
  ArrayRef<int64_t> flattenedExpr = flattener.operandExprStack.back();
  if (!expr.isPureAffine() &&
      expr == getAffineExprFromFlatForm(flattenedExpr, numDims, numSymbols,
                                        flattener.localExprs,
                                        expr.getContext()))
    return expr;
  AffineExpr simplifiedExpr =
      expr.isPureAffine()
          ? getAffineExprFromFlatForm(flattenedExpr, numDims, numSymbols,
                                      flattener.localExprs, expr.getContext())
          : getSemiAffineExprFromFlatForm(flattenedExpr, numDims, numSymbols,
                                          flattener.localExprs,
                                          expr.getContext());

  flattener.operandExprStack.pop_back();
  assert(flattener.operandExprStack.empty());
  return simplifiedExpr;
}
