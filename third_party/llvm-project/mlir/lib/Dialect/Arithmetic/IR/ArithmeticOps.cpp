//===- ArithmeticOps.cpp - MLIR Arithmetic dialect ops implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/APSInt.h"

using namespace mlir;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// Pattern helpers
//===----------------------------------------------------------------------===//

static IntegerAttr addIntegerAttrs(PatternRewriter &builder, Value res,
                                   Attribute lhs, Attribute rhs) {
  return builder.getIntegerAttr(res.getType(),
                                lhs.cast<IntegerAttr>().getInt() +
                                    rhs.cast<IntegerAttr>().getInt());
}

static IntegerAttr subIntegerAttrs(PatternRewriter &builder, Value res,
                                   Attribute lhs, Attribute rhs) {
  return builder.getIntegerAttr(res.getType(),
                                lhs.cast<IntegerAttr>().getInt() -
                                    rhs.cast<IntegerAttr>().getInt());
}

/// Invert an integer comparison predicate.
arith::CmpIPredicate arith::invertPredicate(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return arith::CmpIPredicate::ne;
  case arith::CmpIPredicate::ne:
    return arith::CmpIPredicate::eq;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ule;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ult;
  }
  llvm_unreachable("unknown cmpi predicate kind");
}

static arith::CmpIPredicateAttr invertPredicate(arith::CmpIPredicateAttr pred) {
  return arith::CmpIPredicateAttr::get(pred.getContext(),
                                       invertPredicate(pred.getValue()));
}

//===----------------------------------------------------------------------===//
// TableGen'd canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
#include "ArithmeticCanonicalization.inc"
} // namespace

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void arith::ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto type = getType();
  if (auto intCst = getValue().dyn_cast<IntegerAttr>()) {
    auto intType = type.dyn_cast<IntegerType>();

    // Sugar i1 constants with 'true' and 'false'.
    if (intType && intType.getWidth() == 1)
      return setNameFn(getResult(), (intCst.getInt() ? "true" : "false"));

    // Otherwise, build a compex name with the value and type.
    SmallString<32> specialNameBuffer;
    llvm::raw_svector_ostream specialName(specialNameBuffer);
    specialName << 'c' << intCst.getInt();
    if (intType)
      specialName << '_' << type;
    setNameFn(getResult(), specialName.str());
  } else {
    setNameFn(getResult(), "cst");
  }
}

/// TODO: disallow arith.constant to return anything other than signless integer
/// or float like.
static LogicalResult verify(arith::ConstantOp op) {
  auto type = op.getType();
  // The value's type must match the return type.
  if (op.getValue().getType() != type) {
    return op.emitOpError() << "value type " << op.getValue().getType()
                            << " must match return type: " << type;
  }
  // Integer values must be signless.
  if (type.isa<IntegerType>() && !type.cast<IntegerType>().isSignless())
    return op.emitOpError("integer return type must be signless");
  // Any float or elements attribute are acceptable.
  if (!op.getValue().isa<IntegerAttr, FloatAttr, ElementsAttr>()) {
    return op.emitOpError(
        "value must be an integer, float, or elements attribute");
  }
  return success();
}

bool arith::ConstantOp::isBuildableWith(Attribute value, Type type) {
  // The value's type must be the same as the provided type.
  if (value.getType() != type)
    return false;
  // Integer values must be signless.
  if (type.isa<IntegerType>() && !type.cast<IntegerType>().isSignless())
    return false;
  // Integer, float, and element attributes are buildable.
  return value.isa<IntegerAttr, FloatAttr, ElementsAttr>();
}

OpFoldResult arith::ConstantOp::fold(ArrayRef<Attribute> operands) {
  return getValue();
}

void arith::ConstantIntOp::build(OpBuilder &builder, OperationState &result,
                                 int64_t value, unsigned width) {
  auto type = builder.getIntegerType(width);
  arith::ConstantOp::build(builder, result, type,
                           builder.getIntegerAttr(type, value));
}

void arith::ConstantIntOp::build(OpBuilder &builder, OperationState &result,
                                 int64_t value, Type type) {
  assert(type.isSignlessInteger() &&
         "ConstantIntOp can only have signless integer type values");
  arith::ConstantOp::build(builder, result, type,
                           builder.getIntegerAttr(type, value));
}

bool arith::ConstantIntOp::classof(Operation *op) {
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(op))
    return constOp.getType().isSignlessInteger();
  return false;
}

void arith::ConstantFloatOp::build(OpBuilder &builder, OperationState &result,
                                   const APFloat &value, FloatType type) {
  arith::ConstantOp::build(builder, result, type,
                           builder.getFloatAttr(type, value));
}

bool arith::ConstantFloatOp::classof(Operation *op) {
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(op))
    return constOp.getType().isa<FloatType>();
  return false;
}

void arith::ConstantIndexOp::build(OpBuilder &builder, OperationState &result,
                                   int64_t value) {
  arith::ConstantOp::build(builder, result, builder.getIndexType(),
                           builder.getIndexAttr(value));
}

bool arith::ConstantIndexOp::classof(Operation *op) {
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(op))
    return constOp.getType().isIndex();
  return false;
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::AddIOp::fold(ArrayRef<Attribute> operands) {
  // addi(x, 0) -> x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();

  // add(sub(a, b), b) -> a
  if (auto sub = getLhs().getDefiningOp<SubIOp>())
    if (getRhs() == sub.getRhs())
      return sub.getLhs();

  // add(b, sub(a, b)) -> a
  if (auto sub = getRhs().getDefiningOp<SubIOp>())
    if (getLhs() == sub.getRhs())
      return sub.getLhs();

  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, const APInt &b) { return std::move(a) + b; });
}

void arith::AddIOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<AddIAddConstant, AddISubConstantRHS, AddISubConstantLHS>(
      context);
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::SubIOp::fold(ArrayRef<Attribute> operands) {
  // subi(x,x) -> 0
  if (getOperand(0) == getOperand(1))
    return Builder(getContext()).getZeroAttr(getType());
  // subi(x,0) -> x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();

  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, const APInt &b) { return std::move(a) - b; });
}

void arith::SubIOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<SubIRHSAddConstant, SubILHSAddConstant, SubIRHSSubConstantRHS,
                  SubIRHSSubConstantLHS, SubILHSSubConstantRHS,
                  SubILHSSubConstantLHS>(context);
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MulIOp::fold(ArrayRef<Attribute> operands) {
  // muli(x, 0) -> 0
  if (matchPattern(getRhs(), m_Zero()))
    return getRhs();
  // muli(x, 1) -> x
  if (matchPattern(getRhs(), m_One()))
    return getOperand(0);
  // TODO: Handle the overflow case.

  // default folder
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](const APInt &a, const APInt &b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::DivUIOp::fold(ArrayRef<Attribute> operands) {
  // Don't fold if it would require a division by zero.
  bool div0 = false;
  auto result =
      constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, const APInt &b) {
        if (div0 || !b) {
          div0 = true;
          return a;
        }
        return a.udiv(b);
      });

  // Fold out division by one. Assumes all tensors of all ones are splats.
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue() == 1)
      return getLhs();
  } else if (auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>()) {
    if (rhs.getSplatValue<IntegerAttr>().getValue() == 1)
      return getLhs();
  }

  return div0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::DivSIOp::fold(ArrayRef<Attribute> operands) {
  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result =
      constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, const APInt &b) {
        if (overflowOrDiv0 || !b) {
          overflowOrDiv0 = true;
          return a;
        }
        return a.sdiv_ov(b, overflowOrDiv0);
      });

  // Fold out division by one. Assumes all tensors of all ones are splats.
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue() == 1)
      return getLhs();
  } else if (auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>()) {
    if (rhs.getSplatValue<IntegerAttr>().getValue() == 1)
      return getLhs();
  }

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// Ceil and floor division folding helpers
//===----------------------------------------------------------------------===//

static APInt signedCeilNonnegInputs(const APInt &a, const APInt &b,
                                    bool &overflow) {
  // Returns (a-1)/b + 1
  APInt one(a.getBitWidth(), 1, true); // Signed value 1.
  APInt val = a.ssub_ov(one, overflow).sdiv_ov(b, overflow);
  return val.sadd_ov(one, overflow);
}

//===----------------------------------------------------------------------===//
// CeilDivUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::CeilDivUIOp::fold(ArrayRef<Attribute> operands) {
  bool overflowOrDiv0 = false;
  auto result =
      constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, const APInt &b) {
        if (overflowOrDiv0 || !b) {
          overflowOrDiv0 = true;
          return a;
        }
        APInt quotient = a.udiv(b);
        if (!a.urem(b))
          return quotient;
        APInt one(a.getBitWidth(), 1, true);
        return quotient.uadd_ov(one, overflowOrDiv0);
      });
  // Fold out ceil division by one. Assumes all tensors of all ones are
  // splats.
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue() == 1)
      return getLhs();
  } else if (auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>()) {
    if (rhs.getSplatValue<IntegerAttr>().getValue() == 1)
      return getLhs();
  }

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// CeilDivSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::CeilDivSIOp::fold(ArrayRef<Attribute> operands) {
  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result =
      constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, const APInt &b) {
        if (overflowOrDiv0 || !b) {
          overflowOrDiv0 = true;
          return a;
        }
        if (!a)
          return a;
        // After this point we know that neither a or b are zero.
        unsigned bits = a.getBitWidth();
        APInt zero = APInt::getZero(bits);
        bool aGtZero = a.sgt(zero);
        bool bGtZero = b.sgt(zero);
        if (aGtZero && bGtZero) {
          // Both positive, return ceil(a, b).
          return signedCeilNonnegInputs(a, b, overflowOrDiv0);
        }
        if (!aGtZero && !bGtZero) {
          // Both negative, return ceil(-a, -b).
          APInt posA = zero.ssub_ov(a, overflowOrDiv0);
          APInt posB = zero.ssub_ov(b, overflowOrDiv0);
          return signedCeilNonnegInputs(posA, posB, overflowOrDiv0);
        }
        if (!aGtZero && bGtZero) {
          // A is negative, b is positive, return - ( -a / b).
          APInt posA = zero.ssub_ov(a, overflowOrDiv0);
          APInt div = posA.sdiv_ov(b, overflowOrDiv0);
          return zero.ssub_ov(div, overflowOrDiv0);
        }
        // A is positive, b is negative, return - (a / -b).
        APInt posB = zero.ssub_ov(b, overflowOrDiv0);
        APInt div = a.sdiv_ov(posB, overflowOrDiv0);
        return zero.ssub_ov(div, overflowOrDiv0);
      });

  // Fold out ceil division by one. Assumes all tensors of all ones are
  // splats.
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue() == 1)
      return getLhs();
  } else if (auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>()) {
    if (rhs.getSplatValue<IntegerAttr>().getValue() == 1)
      return getLhs();
  }

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// FloorDivSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::FloorDivSIOp::fold(ArrayRef<Attribute> operands) {
  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result =
      constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, const APInt &b) {
        if (overflowOrDiv0 || !b) {
          overflowOrDiv0 = true;
          return a;
        }
        if (!a)
          return a;
        // After this point we know that neither a or b are zero.
        unsigned bits = a.getBitWidth();
        APInt zero = APInt::getZero(bits);
        bool aGtZero = a.sgt(zero);
        bool bGtZero = b.sgt(zero);
        if (aGtZero && bGtZero) {
          // Both positive, return a / b.
          return a.sdiv_ov(b, overflowOrDiv0);
        }
        if (!aGtZero && !bGtZero) {
          // Both negative, return -a / -b.
          APInt posA = zero.ssub_ov(a, overflowOrDiv0);
          APInt posB = zero.ssub_ov(b, overflowOrDiv0);
          return posA.sdiv_ov(posB, overflowOrDiv0);
        }
        if (!aGtZero && bGtZero) {
          // A is negative, b is positive, return - ceil(-a, b).
          APInt posA = zero.ssub_ov(a, overflowOrDiv0);
          APInt ceil = signedCeilNonnegInputs(posA, b, overflowOrDiv0);
          return zero.ssub_ov(ceil, overflowOrDiv0);
        }
        // A is positive, b is negative, return - ceil(a, -b).
        APInt posB = zero.ssub_ov(b, overflowOrDiv0);
        APInt ceil = signedCeilNonnegInputs(a, posB, overflowOrDiv0);
        return zero.ssub_ov(ceil, overflowOrDiv0);
      });

  // Fold out floor division by one. Assumes all tensors of all ones are
  // splats.
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue() == 1)
      return getLhs();
  } else if (auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>()) {
    if (rhs.getSplatValue<IntegerAttr>().getValue() == 1)
      return getLhs();
  }

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::RemUIOp::fold(ArrayRef<Attribute> operands) {
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!rhs)
    return {};
  auto rhsValue = rhs.getValue();

  // x % 1 = 0
  if (rhsValue.isOneValue())
    return IntegerAttr::get(rhs.getType(), APInt(rhsValue.getBitWidth(), 0));

  // Don't fold if it requires division by zero.
  if (rhsValue.isNullValue())
    return {};

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!lhs)
    return {};
  return IntegerAttr::get(lhs.getType(), lhs.getValue().urem(rhsValue));
}

//===----------------------------------------------------------------------===//
// RemSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::RemSIOp::fold(ArrayRef<Attribute> operands) {
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!rhs)
    return {};
  auto rhsValue = rhs.getValue();

  // x % 1 = 0
  if (rhsValue.isOneValue())
    return IntegerAttr::get(rhs.getType(), APInt(rhsValue.getBitWidth(), 0));

  // Don't fold if it requires division by zero.
  if (rhsValue.isNullValue())
    return {};

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!lhs)
    return {};
  return IntegerAttr::get(lhs.getType(), lhs.getValue().srem(rhsValue));
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::AndIOp::fold(ArrayRef<Attribute> operands) {
  /// and(x, 0) -> 0
  if (matchPattern(getRhs(), m_Zero()))
    return getRhs();
  /// and(x, allOnes) -> x
  APInt intValue;
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) && intValue.isAllOnes())
    return getLhs();

  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, const APInt &b) { return std::move(a) & b; });
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::OrIOp::fold(ArrayRef<Attribute> operands) {
  /// or(x, 0) -> x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();
  /// or(x, <all ones>) -> <all ones>
  if (auto rhsAttr = operands[1].dyn_cast_or_null<IntegerAttr>())
    if (rhsAttr.getValue().isAllOnes())
      return rhsAttr;

  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, const APInt &b) { return std::move(a) | b; });
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::XOrIOp::fold(ArrayRef<Attribute> operands) {
  /// xor(x, 0) -> x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();
  /// xor(x, x) -> 0
  if (getLhs() == getRhs())
    return Builder(getContext()).getZeroAttr(getType());
  /// xor(xor(x, a), a) -> x
  if (arith::XOrIOp prev = getLhs().getDefiningOp<arith::XOrIOp>())
    if (prev.getRhs() == getRhs())
      return prev.getLhs();

  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, const APInt &b) { return std::move(a) ^ b; });
}

void arith::XOrIOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<XOrINotCmpI>(context);
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::AddFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](const APFloat &a, const APFloat &b) { return a + b; });
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::SubFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](const APFloat &a, const APFloat &b) { return a - b; });
}

//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

OpFoldResult MaxSIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  // maxsi(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  APInt intValue;
  // maxsi(x,MAX_INT) -> MAX_INT
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) &&
      intValue.isMaxSignedValue())
    return getRhs();

  // maxsi(x, MIN_INT) -> x
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) &&
      intValue.isMinSignedValue())
    return getLhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](const APInt &a, const APInt &b) {
                                          return llvm::APIntOps::smax(a, b);
                                        });
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

OpFoldResult MaxUIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  // maxui(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  APInt intValue;
  // maxui(x,MAX_INT) -> MAX_INT
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) && intValue.isMaxValue())
    return getRhs();

  // maxui(x, MIN_INT) -> x
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) && intValue.isMinValue())
    return getLhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](const APInt &a, const APInt &b) {
                                          return llvm::APIntOps::umax(a, b);
                                        });
}

//===----------------------------------------------------------------------===//
// MinSIOp
//===----------------------------------------------------------------------===//

OpFoldResult MinSIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  // minsi(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  APInt intValue;
  // minsi(x,MIN_INT) -> MIN_INT
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) &&
      intValue.isMinSignedValue())
    return getRhs();

  // minsi(x, MAX_INT) -> x
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) &&
      intValue.isMaxSignedValue())
    return getLhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](const APInt &a, const APInt &b) {
                                          return llvm::APIntOps::smin(a, b);
                                        });
}

//===----------------------------------------------------------------------===//
// MinUIOp
//===----------------------------------------------------------------------===//

OpFoldResult MinUIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  // minui(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  APInt intValue;
  // minui(x,MIN_INT) -> MIN_INT
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) && intValue.isMinValue())
    return getRhs();

  // minui(x, MAX_INT) -> x
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) && intValue.isMaxValue())
    return getLhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](const APInt &a, const APInt &b) {
                                          return llvm::APIntOps::umin(a, b);
                                        });
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MulFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](const APFloat &a, const APFloat &b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::DivFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](const APFloat &a, const APFloat &b) { return a / b; });
}

//===----------------------------------------------------------------------===//
// Utility functions for verifying cast ops
//===----------------------------------------------------------------------===//

template <typename... Types>
using type_list = std::tuple<Types...> *;

/// Returns a non-null type only if the provided type is one of the allowed
/// types or one of the allowed shaped types of the allowed types. Returns the
/// element type if a valid shaped type is provided.
template <typename... ShapedTypes, typename... ElementTypes>
static Type getUnderlyingType(Type type, type_list<ShapedTypes...>,
                              type_list<ElementTypes...>) {
  if (type.isa<ShapedType>() && !type.isa<ShapedTypes...>())
    return {};

  auto underlyingType = getElementTypeOrSelf(type);
  if (!underlyingType.isa<ElementTypes...>())
    return {};

  return underlyingType;
}

/// Get allowed underlying types for vectors and tensors.
template <typename... ElementTypes>
static Type getTypeIfLike(Type type) {
  return getUnderlyingType(type, type_list<VectorType, TensorType>(),
                           type_list<ElementTypes...>());
}

/// Get allowed underlying types for vectors, tensors, and memrefs.
template <typename... ElementTypes>
static Type getTypeIfLikeOrMemRef(Type type) {
  return getUnderlyingType(type,
                           type_list<VectorType, TensorType, MemRefType>(),
                           type_list<ElementTypes...>());
}

static bool areValidCastInputsAndOutputs(TypeRange inputs, TypeRange outputs) {
  return inputs.size() == 1 && outputs.size() == 1 &&
         succeeded(verifyCompatibleShapes(inputs.front(), outputs.front()));
}

//===----------------------------------------------------------------------===//
// Verifiers for integer and floating point extension/truncation ops
//===----------------------------------------------------------------------===//

// Extend ops can only extend to a wider type.
template <typename ValType, typename Op>
static LogicalResult verifyExtOp(Op op) {
  Type srcType = getElementTypeOrSelf(op.getIn().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (srcType.cast<ValType>().getWidth() >= dstType.cast<ValType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be wider than operand type " << srcType;

  return success();
}

// Truncate ops can only truncate to a shorter type.
template <typename ValType, typename Op>
static LogicalResult verifyTruncateOp(Op op) {
  Type srcType = getElementTypeOrSelf(op.getIn().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (srcType.cast<ValType>().getWidth() <= dstType.cast<ValType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be shorter than operand type " << srcType;

  return success();
}

/// Validate a cast that changes the width of a type.
template <template <typename> class WidthComparator, typename... ElementTypes>
static bool checkWidthChangeCast(TypeRange inputs, TypeRange outputs) {
  if (!areValidCastInputsAndOutputs(inputs, outputs))
    return false;

  auto srcType = getTypeIfLike<ElementTypes...>(inputs.front());
  auto dstType = getTypeIfLike<ElementTypes...>(outputs.front());
  if (!srcType || !dstType)
    return false;

  return WidthComparator<unsigned>()(dstType.getIntOrFloatBitWidth(),
                                     srcType.getIntOrFloatBitWidth());
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::ExtUIOp::fold(ArrayRef<Attribute> operands) {
  if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>())
    return IntegerAttr::get(
        getType(), lhs.getValue().zext(getType().getIntOrFloatBitWidth()));

  if (auto lhs = getIn().getDefiningOp<ExtUIOp>()) {
    getInMutable().assign(lhs.getIn());
    return getResult();
  }

  return {};
}

bool arith::ExtUIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkWidthChangeCast<std::greater, IntegerType>(inputs, outputs);
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::ExtSIOp::fold(ArrayRef<Attribute> operands) {
  if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>())
    return IntegerAttr::get(
        getType(), lhs.getValue().sext(getType().getIntOrFloatBitWidth()));

  if (auto lhs = getIn().getDefiningOp<ExtSIOp>()) {
    getInMutable().assign(lhs.getIn());
    return getResult();
  }

  return {};
}

bool arith::ExtSIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkWidthChangeCast<std::greater, IntegerType>(inputs, outputs);
}

void arith::ExtSIOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<ExtSIOfExtUI>(context);
}

//===----------------------------------------------------------------------===//
// ExtFOp
//===----------------------------------------------------------------------===//

bool arith::ExtFOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkWidthChangeCast<std::greater, FloatType>(inputs, outputs);
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::TruncIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary operation takes one operand");

  // trunci(zexti(a)) -> a
  // trunci(sexti(a)) -> a
  if (matchPattern(getOperand(), m_Op<arith::ExtUIOp>()) ||
      matchPattern(getOperand(), m_Op<arith::ExtSIOp>()))
    return getOperand().getDefiningOp()->getOperand(0);

  // trunci(trunci(a)) -> trunci(a))
  if (matchPattern(getOperand(), m_Op<arith::TruncIOp>())) {
    setOperand(getOperand().getDefiningOp()->getOperand(0));
    return getResult();
  }

  if (!operands[0])
    return {};

  if (auto lhs = operands[0].dyn_cast<IntegerAttr>()) {
    return IntegerAttr::get(
        getType(), lhs.getValue().trunc(getType().getIntOrFloatBitWidth()));
  }

  return {};
}

bool arith::TruncIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkWidthChangeCast<std::less, IntegerType>(inputs, outputs);
}

//===----------------------------------------------------------------------===//
// TruncFOp
//===----------------------------------------------------------------------===//

/// Perform safe const propagation for truncf, i.e. only propagate if FP value
/// can be represented without precision loss or rounding.
OpFoldResult arith::TruncFOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary operation takes one operand");

  auto constOperand = operands.front();
  if (!constOperand || !constOperand.isa<FloatAttr>())
    return {};

  // Convert to target type via 'double'.
  double sourceValue =
      constOperand.dyn_cast<FloatAttr>().getValue().convertToDouble();
  auto targetAttr = FloatAttr::get(getType(), sourceValue);

  // Propagate if constant's value does not change after truncation.
  if (sourceValue == targetAttr.getValue().convertToDouble())
    return targetAttr;

  return {};
}

bool arith::TruncFOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkWidthChangeCast<std::less, FloatType>(inputs, outputs);
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

void arith::AndIOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<AndOfExtUI, AndOfExtSI>(context);
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

void arith::OrIOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<OrOfExtUI, OrOfExtSI>(context);
}

//===----------------------------------------------------------------------===//
// Verifiers for casts between integers and floats.
//===----------------------------------------------------------------------===//

template <typename From, typename To>
static bool checkIntFloatCast(TypeRange inputs, TypeRange outputs) {
  if (!areValidCastInputsAndOutputs(inputs, outputs))
    return false;

  auto srcType = getTypeIfLike<From>(inputs.front());
  auto dstType = getTypeIfLike<To>(outputs.back());

  return srcType && dstType;
}

//===----------------------------------------------------------------------===//
// UIToFPOp
//===----------------------------------------------------------------------===//

bool arith::UIToFPOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkIntFloatCast<IntegerType, FloatType>(inputs, outputs);
}

OpFoldResult arith::UIToFPOp::fold(ArrayRef<Attribute> operands) {
  if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    const APInt &api = lhs.getValue();
    FloatType floatTy = getType().cast<FloatType>();
    APFloat apf(floatTy.getFloatSemantics(),
                APInt::getZero(floatTy.getWidth()));
    apf.convertFromAPInt(api, /*IsSigned=*/false, APFloat::rmNearestTiesToEven);
    return FloatAttr::get(floatTy, apf);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// SIToFPOp
//===----------------------------------------------------------------------===//

bool arith::SIToFPOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkIntFloatCast<IntegerType, FloatType>(inputs, outputs);
}

OpFoldResult arith::SIToFPOp::fold(ArrayRef<Attribute> operands) {
  if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    const APInt &api = lhs.getValue();
    FloatType floatTy = getType().cast<FloatType>();
    APFloat apf(floatTy.getFloatSemantics(),
                APInt::getZero(floatTy.getWidth()));
    apf.convertFromAPInt(api, /*IsSigned=*/true, APFloat::rmNearestTiesToEven);
    return FloatAttr::get(floatTy, apf);
  }
  return {};
}
//===----------------------------------------------------------------------===//
// FPToUIOp
//===----------------------------------------------------------------------===//

bool arith::FPToUIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkIntFloatCast<FloatType, IntegerType>(inputs, outputs);
}

OpFoldResult arith::FPToUIOp::fold(ArrayRef<Attribute> operands) {
  if (auto lhs = operands[0].dyn_cast_or_null<FloatAttr>()) {
    const APFloat &apf = lhs.getValue();
    IntegerType intTy = getType().cast<IntegerType>();
    bool ignored;
    APSInt api(intTy.getWidth(), /*isUnsigned=*/true);
    if (APFloat::opInvalidOp ==
        apf.convertToInteger(api, APFloat::rmTowardZero, &ignored)) {
      // Undefined behavior invoked - the destination type can't represent
      // the input constant.
      return {};
    }
    return IntegerAttr::get(getType(), api);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// FPToSIOp
//===----------------------------------------------------------------------===//

bool arith::FPToSIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkIntFloatCast<FloatType, IntegerType>(inputs, outputs);
}

OpFoldResult arith::FPToSIOp::fold(ArrayRef<Attribute> operands) {
  if (auto lhs = operands[0].dyn_cast_or_null<FloatAttr>()) {
    const APFloat &apf = lhs.getValue();
    IntegerType intTy = getType().cast<IntegerType>();
    bool ignored;
    APSInt api(intTy.getWidth(), /*isUnsigned=*/false);
    if (APFloat::opInvalidOp ==
        apf.convertToInteger(api, APFloat::rmTowardZero, &ignored)) {
      // Undefined behavior invoked - the destination type can't represent
      // the input constant.
      return {};
    }
    return IntegerAttr::get(getType(), api);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

bool arith::IndexCastOp::areCastCompatible(TypeRange inputs,
                                           TypeRange outputs) {
  if (!areValidCastInputsAndOutputs(inputs, outputs))
    return false;

  auto srcType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(inputs.front());
  auto dstType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(outputs.front());
  if (!srcType || !dstType)
    return false;

  return (srcType.isIndex() && dstType.isSignlessInteger()) ||
         (srcType.isSignlessInteger() && dstType.isIndex());
}

OpFoldResult arith::IndexCastOp::fold(ArrayRef<Attribute> operands) {
  // index_cast(constant) -> constant
  // A little hack because we go through int. Otherwise, the size of the
  // constant might need to change.
  if (auto value = operands[0].dyn_cast_or_null<IntegerAttr>())
    return IntegerAttr::get(getType(), value.getInt());

  return {};
}

void arith::IndexCastOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<IndexCastOfIndexCast, IndexCastOfExtSI>(context);
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

bool arith::BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (!areValidCastInputsAndOutputs(inputs, outputs))
    return false;

  auto srcType =
      getTypeIfLikeOrMemRef<IntegerType, IndexType, FloatType>(inputs.front());
  auto dstType =
      getTypeIfLikeOrMemRef<IntegerType, IndexType, FloatType>(outputs.front());
  if (!srcType || !dstType)
    return false;

  return srcType.getIntOrFloatBitWidth() == dstType.getIntOrFloatBitWidth();
}

OpFoldResult arith::BitcastOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "bitcast op expects 1 operand");

  auto resType = getType();
  auto operand = operands[0];
  if (!operand)
    return {};

  /// Bitcast dense elements.
  if (auto denseAttr = operand.dyn_cast_or_null<DenseElementsAttr>())
    return denseAttr.bitcast(resType.cast<ShapedType>().getElementType());
  /// Other shaped types unhandled.
  if (resType.isa<ShapedType>())
    return {};

  /// Bitcast integer or float to integer or float.
  APInt bits = operand.isa<FloatAttr>()
                   ? operand.cast<FloatAttr>().getValue().bitcastToAPInt()
                   : operand.cast<IntegerAttr>().getValue();

  if (auto resFloatType = resType.dyn_cast<FloatType>())
    return FloatAttr::get(resType,
                          APFloat(resFloatType.getFloatSemantics(), bits));
  return IntegerAttr::get(resType, bits);
}

void arith::BitcastOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<BitcastOfBitcast>(context);
}

//===----------------------------------------------------------------------===//
// Helpers for compare ops
//===----------------------------------------------------------------------===//

/// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i1Type);
  if (type.isa<UnrankedTensorType>())
    return UnrankedTensorType::get(i1Type);
  if (auto vectorType = type.dyn_cast<VectorType>())
    return VectorType::get(vectorType.getShape(), i1Type,
                           vectorType.getNumScalableDims());
  return i1Type;
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known integer
/// comparison predicates.
bool mlir::arith::applyCmpPredicate(arith::CmpIPredicate predicate,
                                    const APInt &lhs, const APInt &rhs) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return lhs.eq(rhs);
  case arith::CmpIPredicate::ne:
    return lhs.ne(rhs);
  case arith::CmpIPredicate::slt:
    return lhs.slt(rhs);
  case arith::CmpIPredicate::sle:
    return lhs.sle(rhs);
  case arith::CmpIPredicate::sgt:
    return lhs.sgt(rhs);
  case arith::CmpIPredicate::sge:
    return lhs.sge(rhs);
  case arith::CmpIPredicate::ult:
    return lhs.ult(rhs);
  case arith::CmpIPredicate::ule:
    return lhs.ule(rhs);
  case arith::CmpIPredicate::ugt:
    return lhs.ugt(rhs);
  case arith::CmpIPredicate::uge:
    return lhs.uge(rhs);
  }
  llvm_unreachable("unknown cmpi predicate kind");
}

/// Returns true if the predicate is true for two equal operands.
static bool applyCmpPredicateToEqualOperands(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::uge:
    return true;
  case arith::CmpIPredicate::ne:
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ugt:
    return false;
  }
  llvm_unreachable("unknown cmpi predicate kind");
}

static Attribute getBoolAttribute(Type type, MLIRContext *ctx, bool value) {
  auto boolAttr = BoolAttr::get(ctx, value);
  ShapedType shapedType = type.dyn_cast_or_null<ShapedType>();
  if (!shapedType)
    return boolAttr;
  return DenseElementsAttr::get(shapedType, boolAttr);
}

OpFoldResult arith::CmpIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "cmpi takes two operands");

  // cmpi(pred, x, x)
  if (getLhs() == getRhs()) {
    auto val = applyCmpPredicateToEqualOperands(getPredicate());
    return getBoolAttribute(getType(), getContext(), val);
  }

  if (matchPattern(getRhs(), m_Zero())) {
    if (auto extOp = getLhs().getDefiningOp<ExtSIOp>()) {
      if (extOp.getOperand().getType().cast<IntegerType>().getWidth() == 1) {
        // extsi(%x : i1 -> iN) != 0  ->  %x
        if (getPredicate() == arith::CmpIPredicate::ne) {
          return extOp.getOperand();
        }
      }
    }
    if (auto extOp = getLhs().getDefiningOp<ExtUIOp>()) {
      if (extOp.getOperand().getType().cast<IntegerType>().getWidth() == 1) {
        // extui(%x : i1 -> iN) != 0  ->  %x
        if (getPredicate() == arith::CmpIPredicate::ne) {
          return extOp.getOperand();
        }
      }
    }
  }

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!lhs || !rhs)
    return {};

  auto val = applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
  return BoolAttr::get(getContext(), val);
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known floating point
/// comparison predicates.
bool mlir::arith::applyCmpPredicate(arith::CmpFPredicate predicate,
                                    const APFloat &lhs, const APFloat &rhs) {
  auto cmpResult = lhs.compare(rhs);
  switch (predicate) {
  case arith::CmpFPredicate::AlwaysFalse:
    return false;
  case arith::CmpFPredicate::OEQ:
    return cmpResult == APFloat::cmpEqual;
  case arith::CmpFPredicate::OGT:
    return cmpResult == APFloat::cmpGreaterThan;
  case arith::CmpFPredicate::OGE:
    return cmpResult == APFloat::cmpGreaterThan ||
           cmpResult == APFloat::cmpEqual;
  case arith::CmpFPredicate::OLT:
    return cmpResult == APFloat::cmpLessThan;
  case arith::CmpFPredicate::OLE:
    return cmpResult == APFloat::cmpLessThan || cmpResult == APFloat::cmpEqual;
  case arith::CmpFPredicate::ONE:
    return cmpResult != APFloat::cmpUnordered && cmpResult != APFloat::cmpEqual;
  case arith::CmpFPredicate::ORD:
    return cmpResult != APFloat::cmpUnordered;
  case arith::CmpFPredicate::UEQ:
    return cmpResult == APFloat::cmpUnordered || cmpResult == APFloat::cmpEqual;
  case arith::CmpFPredicate::UGT:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpGreaterThan;
  case arith::CmpFPredicate::UGE:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpGreaterThan ||
           cmpResult == APFloat::cmpEqual;
  case arith::CmpFPredicate::ULT:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpLessThan;
  case arith::CmpFPredicate::ULE:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpLessThan || cmpResult == APFloat::cmpEqual;
  case arith::CmpFPredicate::UNE:
    return cmpResult != APFloat::cmpEqual;
  case arith::CmpFPredicate::UNO:
    return cmpResult == APFloat::cmpUnordered;
  case arith::CmpFPredicate::AlwaysTrue:
    return true;
  }
  llvm_unreachable("unknown cmpf predicate kind");
}

OpFoldResult arith::CmpFOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "cmpf takes two operands");

  auto lhs = operands.front().dyn_cast_or_null<FloatAttr>();
  auto rhs = operands.back().dyn_cast_or_null<FloatAttr>();

  if (!lhs || !rhs)
    return {};

  auto val = applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
  return BoolAttr::get(getContext(), val);
}

//===----------------------------------------------------------------------===//
// Atomic Enum
//===----------------------------------------------------------------------===//

/// Returns the identity value attribute associated with an AtomicRMWKind op.
Attribute mlir::arith::getIdentityValueAttr(AtomicRMWKind kind, Type resultType,
                                            OpBuilder &builder, Location loc) {
  switch (kind) {
  case AtomicRMWKind::maxf:
    return builder.getFloatAttr(
        resultType,
        APFloat::getInf(resultType.cast<FloatType>().getFloatSemantics(),
                        /*Negative=*/true));
  case AtomicRMWKind::addf:
  case AtomicRMWKind::addi:
  case AtomicRMWKind::maxu:
  case AtomicRMWKind::ori:
    return builder.getZeroAttr(resultType);
  case AtomicRMWKind::andi:
    return builder.getIntegerAttr(
        resultType,
        APInt::getAllOnes(resultType.cast<IntegerType>().getWidth()));
  case AtomicRMWKind::maxs:
    return builder.getIntegerAttr(
        resultType,
        APInt::getSignedMinValue(resultType.cast<IntegerType>().getWidth()));
  case AtomicRMWKind::minf:
    return builder.getFloatAttr(
        resultType,
        APFloat::getInf(resultType.cast<FloatType>().getFloatSemantics(),
                        /*Negative=*/false));
  case AtomicRMWKind::mins:
    return builder.getIntegerAttr(
        resultType,
        APInt::getSignedMaxValue(resultType.cast<IntegerType>().getWidth()));
  case AtomicRMWKind::minu:
    return builder.getIntegerAttr(
        resultType,
        APInt::getMaxValue(resultType.cast<IntegerType>().getWidth()));
  case AtomicRMWKind::muli:
    return builder.getIntegerAttr(resultType, 1);
  case AtomicRMWKind::mulf:
    return builder.getFloatAttr(resultType, 1);
  // TODO: Add remaining reduction operations.
  default:
    (void)emitOptionalError(loc, "Reduction operation type not supported");
    break;
  }
  return nullptr;
}

/// Returns the identity value associated with an AtomicRMWKind op.
Value mlir::arith::getIdentityValue(AtomicRMWKind op, Type resultType,
                                    OpBuilder &builder, Location loc) {
  Attribute attr = getIdentityValueAttr(op, resultType, builder, loc);
  return builder.create<arith::ConstantOp>(loc, attr);
}

/// Return the value obtained by applying the reduction operation kind
/// associated with a binary AtomicRMWKind op to `lhs` and `rhs`.
Value mlir::arith::getReductionOp(AtomicRMWKind op, OpBuilder &builder,
                                  Location loc, Value lhs, Value rhs) {
  switch (op) {
  case AtomicRMWKind::addf:
    return builder.create<arith::AddFOp>(loc, lhs, rhs);
  case AtomicRMWKind::addi:
    return builder.create<arith::AddIOp>(loc, lhs, rhs);
  case AtomicRMWKind::mulf:
    return builder.create<arith::MulFOp>(loc, lhs, rhs);
  case AtomicRMWKind::muli:
    return builder.create<arith::MulIOp>(loc, lhs, rhs);
  case AtomicRMWKind::maxf:
    return builder.create<arith::MaxFOp>(loc, lhs, rhs);
  case AtomicRMWKind::minf:
    return builder.create<arith::MinFOp>(loc, lhs, rhs);
  case AtomicRMWKind::maxs:
    return builder.create<arith::MaxSIOp>(loc, lhs, rhs);
  case AtomicRMWKind::mins:
    return builder.create<arith::MinSIOp>(loc, lhs, rhs);
  case AtomicRMWKind::maxu:
    return builder.create<arith::MaxUIOp>(loc, lhs, rhs);
  case AtomicRMWKind::minu:
    return builder.create<arith::MinUIOp>(loc, lhs, rhs);
  case AtomicRMWKind::ori:
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  case AtomicRMWKind::andi:
    return builder.create<arith::AndIOp>(loc, lhs, rhs);
  // TODO: Add remaining reduction operations.
  default:
    (void)emitOptionalError(loc, "Reduction operation type not supported");
    break;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd enum attribute definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/ArithmeticOpsEnums.cpp.inc"
