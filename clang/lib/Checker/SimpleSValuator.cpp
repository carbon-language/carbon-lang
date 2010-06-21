// SimpleSValuator.cpp - A basic SValuator ------------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SimpleSValuator, a basic implementation of SValuator.
//
//===----------------------------------------------------------------------===//

#include "clang/Checker/PathSensitive/SValuator.h"
#include "clang/Checker/PathSensitive/GRState.h"

using namespace clang;

namespace {
class SimpleSValuator : public SValuator {
protected:
  virtual SVal EvalCastNL(NonLoc val, QualType castTy);
  virtual SVal EvalCastL(Loc val, QualType castTy);

public:
  SimpleSValuator(ValueManager &valMgr) : SValuator(valMgr) {}
  virtual ~SimpleSValuator() {}

  virtual SVal EvalMinus(NonLoc val);
  virtual SVal EvalComplement(NonLoc val);
  virtual SVal EvalBinOpNN(const GRState *state, BinaryOperator::Opcode op,
                           NonLoc lhs, NonLoc rhs, QualType resultTy);
  virtual SVal EvalBinOpLL(BinaryOperator::Opcode op, Loc lhs, Loc rhs,
                           QualType resultTy);
  virtual SVal EvalBinOpLN(const GRState *state, BinaryOperator::Opcode op,
                           Loc lhs, NonLoc rhs, QualType resultTy);
  
  SVal MakeSymIntVal(const SymExpr *LHS, BinaryOperator::Opcode op,
                     const llvm::APSInt &RHS, QualType resultTy);
};
} // end anonymous namespace

SValuator *clang::CreateSimpleSValuator(ValueManager &valMgr) {
  return new SimpleSValuator(valMgr);
}

//===----------------------------------------------------------------------===//
// Transfer function for Casts.
//===----------------------------------------------------------------------===//

SVal SimpleSValuator::EvalCastNL(NonLoc val, QualType castTy) {

  bool isLocType = Loc::IsLocType(castTy);

  if (nonloc::LocAsInteger *LI = dyn_cast<nonloc::LocAsInteger>(&val)) {
    if (isLocType)
      return LI->getLoc();

    // FIXME: Correctly support promotions/truncations.
    ASTContext &Ctx = ValMgr.getContext();
    unsigned castSize = Ctx.getTypeSize(castTy);
    if (castSize == LI->getNumBits())
      return val;

    return ValMgr.makeLocAsInteger(LI->getLoc(), castSize);
  }

  if (const SymExpr *se = val.getAsSymbolicExpression()) {
    ASTContext &Ctx = ValMgr.getContext();
    QualType T = Ctx.getCanonicalType(se->getType(Ctx));
    if (T == Ctx.getCanonicalType(castTy))
      return val;
    
    // FIXME: Remove this hack when we support symbolic truncation/extension.
    // HACK: If both castTy and T are integers, ignore the cast.  This is
    // not a permanent solution.  Eventually we want to precisely handle
    // extension/truncation of symbolic integers.  This prevents us from losing
    // precision when we assign 'x = y' and 'y' is symbolic and x and y are
    // different integer types.
    if (T->isIntegerType() && castTy->isIntegerType())
      return val;

    return UnknownVal();
  }

  if (!isa<nonloc::ConcreteInt>(val))
    return UnknownVal();

  // Only handle casts from integers to integers.
  if (!isLocType && !castTy->isIntegerType())
    return UnknownVal();

  llvm::APSInt i = cast<nonloc::ConcreteInt>(val).getValue();
  i.setIsUnsigned(castTy->isUnsignedIntegerType() || Loc::IsLocType(castTy));
  i.extOrTrunc(ValMgr.getContext().getTypeSize(castTy));

  if (isLocType)
    return ValMgr.makeIntLocVal(i);
  else
    return ValMgr.makeIntVal(i);
}

SVal SimpleSValuator::EvalCastL(Loc val, QualType castTy) {

  // Casts from pointers -> pointers, just return the lval.
  //
  // Casts from pointers -> references, just return the lval.  These
  //   can be introduced by the frontend for corner cases, e.g
  //   casting from va_list* to __builtin_va_list&.
  //
  if (Loc::IsLocType(castTy) || castTy->isReferenceType())
    return val;

  // FIXME: Handle transparent unions where a value can be "transparently"
  //  lifted into a union type.
  if (castTy->isUnionType())
    return UnknownVal();

  if (castTy->isIntegerType()) {
    unsigned BitWidth = ValMgr.getContext().getTypeSize(castTy);

    if (!isa<loc::ConcreteInt>(val))
      return ValMgr.makeLocAsInteger(val, BitWidth);

    llvm::APSInt i = cast<loc::ConcreteInt>(val).getValue();
    i.setIsUnsigned(castTy->isUnsignedIntegerType() || Loc::IsLocType(castTy));
    i.extOrTrunc(BitWidth);
    return ValMgr.makeIntVal(i);
  }

  // All other cases: return 'UnknownVal'.  This includes casting pointers
  // to floats, which is probably badness it itself, but this is a good
  // intermediate solution until we do something better.
  return UnknownVal();
}

//===----------------------------------------------------------------------===//
// Transfer function for unary operators.
//===----------------------------------------------------------------------===//

SVal SimpleSValuator::EvalMinus(NonLoc val) {
  switch (val.getSubKind()) {
  case nonloc::ConcreteIntKind:
    return cast<nonloc::ConcreteInt>(val).evalMinus(ValMgr);
  default:
    return UnknownVal();
  }
}

SVal SimpleSValuator::EvalComplement(NonLoc X) {
  switch (X.getSubKind()) {
  case nonloc::ConcreteIntKind:
    return cast<nonloc::ConcreteInt>(X).evalComplement(ValMgr);
  default:
    return UnknownVal();
  }
}

//===----------------------------------------------------------------------===//
// Transfer function for binary operators.
//===----------------------------------------------------------------------===//

static BinaryOperator::Opcode NegateComparison(BinaryOperator::Opcode op) {
  switch (op) {
  default:
    assert(false && "Invalid opcode.");
  case BinaryOperator::LT: return BinaryOperator::GE;
  case BinaryOperator::GT: return BinaryOperator::LE;
  case BinaryOperator::LE: return BinaryOperator::GT;
  case BinaryOperator::GE: return BinaryOperator::LT;
  case BinaryOperator::EQ: return BinaryOperator::NE;
  case BinaryOperator::NE: return BinaryOperator::EQ;
  }
}

// Equality operators for Locs.
// FIXME: All this logic will be revamped when we have MemRegion::getLocation()
// implemented.

static SVal EvalEquality(ValueManager &ValMgr, Loc lhs, Loc rhs, bool isEqual,
                         QualType resultTy) {

  switch (lhs.getSubKind()) {
    default:
      assert(false && "EQ/NE not implemented for this Loc.");
      return UnknownVal();

    case loc::ConcreteIntKind: {
      if (SymbolRef rSym = rhs.getAsSymbol())
        return ValMgr.makeNonLoc(rSym,
                                 isEqual ? BinaryOperator::EQ
                                 : BinaryOperator::NE,
                                 cast<loc::ConcreteInt>(lhs).getValue(),
                                 resultTy);
      break;
    }
    case loc::MemRegionKind: {
      if (SymbolRef lSym = lhs.getAsLocSymbol()) {
        if (isa<loc::ConcreteInt>(rhs)) {
          return ValMgr.makeNonLoc(lSym,
                                   isEqual ? BinaryOperator::EQ
                                   : BinaryOperator::NE,
                                   cast<loc::ConcreteInt>(rhs).getValue(),
                                   resultTy);
        }
      }
      break;
    }

    case loc::GotoLabelKind:
      break;
  }

  return ValMgr.makeTruthVal(isEqual ? lhs == rhs : lhs != rhs, resultTy);
}

SVal SimpleSValuator::MakeSymIntVal(const SymExpr *LHS,
                                    BinaryOperator::Opcode op,
                                    const llvm::APSInt &RHS,
                                    QualType resultTy) {
  bool isIdempotent = false;

  // Check for a few special cases with known reductions first.
  switch (op) {
  default:
    // We can't reduce this case; just treat it normally.
    break;
  case BinaryOperator::Mul:
    // a*0 and a*1
    if (RHS == 0)
      return ValMgr.makeIntVal(0, resultTy);
    else if (RHS == 1)
      isIdempotent = true;
    break;
  case BinaryOperator::Div:
    // a/0 and a/1
    if (RHS == 0)
      // This is also handled elsewhere.
      return UndefinedVal();
    else if (RHS == 1)
      isIdempotent = true;
    break;
  case BinaryOperator::Rem:
    // a%0 and a%1
    if (RHS == 0)
      // This is also handled elsewhere.
      return UndefinedVal();
    else if (RHS == 1)
      return ValMgr.makeIntVal(0, resultTy);
    break;
  case BinaryOperator::Add:
  case BinaryOperator::Sub:
  case BinaryOperator::Shl:
  case BinaryOperator::Shr:
  case BinaryOperator::Xor:
    // a+0, a-0, a<<0, a>>0, a^0
    if (RHS == 0)
      isIdempotent = true;
    break;
  case BinaryOperator::And:
    // a&0 and a&(~0)
    if (RHS == 0)
      return ValMgr.makeIntVal(0, resultTy);
    else if (RHS.isAllOnesValue())
      isIdempotent = true;
    break;
  case BinaryOperator::Or:
    // a|0 and a|(~0)
    if (RHS == 0)
      isIdempotent = true;
    else if (RHS.isAllOnesValue()) {
      BasicValueFactory &BVF = ValMgr.getBasicValueFactory();
      const llvm::APSInt &Result = BVF.Convert(resultTy, RHS);
      return nonloc::ConcreteInt(Result);
    }
    break;
  }

  // Idempotent ops (like a*1) can still change the type of an expression.
  // Wrap the LHS up in a NonLoc again and let EvalCastNL do the dirty work.
  if (isIdempotent) {
    if (SymbolRef LHSSym = dyn_cast<SymbolData>(LHS))
      return EvalCastNL(nonloc::SymbolVal(LHSSym), resultTy);
    return EvalCastNL(nonloc::SymExprVal(LHS), resultTy);
  }

  // If we reach this point, the expression cannot be simplified.
  // Make a SymExprVal for the entire thing.
  return ValMgr.makeNonLoc(LHS, op, RHS, resultTy);
}

SVal SimpleSValuator::EvalBinOpNN(const GRState *state,
                                  BinaryOperator::Opcode op,
                                  NonLoc lhs, NonLoc rhs,
                                  QualType resultTy)  {
  // Handle trivial case where left-side and right-side are the same.
  if (lhs == rhs)
    switch (op) {
      default:
        break;
      case BinaryOperator::EQ:
      case BinaryOperator::LE:
      case BinaryOperator::GE:
        return ValMgr.makeTruthVal(true, resultTy);
      case BinaryOperator::LT:
      case BinaryOperator::GT:
      case BinaryOperator::NE:
        return ValMgr.makeTruthVal(false, resultTy);
      case BinaryOperator::Xor:
      case BinaryOperator::Sub:
        return ValMgr.makeIntVal(0, resultTy);
      case BinaryOperator::Or:
      case BinaryOperator::And:
        return EvalCastNL(lhs, resultTy);
    }

  while (1) {
    switch (lhs.getSubKind()) {
    default:
      return UnknownVal();
    case nonloc::LocAsIntegerKind: {
      Loc lhsL = cast<nonloc::LocAsInteger>(lhs).getLoc();
      switch (rhs.getSubKind()) {
        case nonloc::LocAsIntegerKind:
          return EvalBinOpLL(op, lhsL, cast<nonloc::LocAsInteger>(rhs).getLoc(),
                             resultTy);
        case nonloc::ConcreteIntKind: {
          // Transform the integer into a location and compare.
          ASTContext& Ctx = ValMgr.getContext();
          llvm::APSInt i = cast<nonloc::ConcreteInt>(rhs).getValue();
          i.setIsUnsigned(true);
          i.extOrTrunc(Ctx.getTypeSize(Ctx.VoidPtrTy));
          return EvalBinOpLL(op, lhsL, ValMgr.makeLoc(i), resultTy);
        }
        default:
          switch (op) {
            case BinaryOperator::EQ:
              return ValMgr.makeTruthVal(false, resultTy);
            case BinaryOperator::NE:
              return ValMgr.makeTruthVal(true, resultTy);
            default:
              // This case also handles pointer arithmetic.
              return UnknownVal();
          }
      }
    }
    case nonloc::SymExprValKind: {
      nonloc::SymExprVal *selhs = cast<nonloc::SymExprVal>(&lhs);

      // Only handle LHS of the form "$sym op constant", at least for now.
      const SymIntExpr *symIntExpr =
        dyn_cast<SymIntExpr>(selhs->getSymbolicExpression());

      if (!symIntExpr)
        return UnknownVal();

      // Is this a logical not? (!x is represented as x == 0.)
      if (op == BinaryOperator::EQ && rhs.isZeroConstant()) {
        // We know how to negate certain expressions. Simplify them here.

        BinaryOperator::Opcode opc = symIntExpr->getOpcode();
        switch (opc) {
        default:
          // We don't know how to negate this operation.
          // Just handle it as if it were a normal comparison to 0.
          break;
        case BinaryOperator::LAnd:
        case BinaryOperator::LOr:
          assert(false && "Logical operators handled by branching logic.");
          return UnknownVal();
        case BinaryOperator::Assign:
        case BinaryOperator::MulAssign:
        case BinaryOperator::DivAssign:
        case BinaryOperator::RemAssign:
        case BinaryOperator::AddAssign:
        case BinaryOperator::SubAssign:
        case BinaryOperator::ShlAssign:
        case BinaryOperator::ShrAssign:
        case BinaryOperator::AndAssign:
        case BinaryOperator::XorAssign:
        case BinaryOperator::OrAssign:
        case BinaryOperator::Comma:
          assert(false && "'=' and ',' operators handled by GRExprEngine.");
          return UnknownVal();
        case BinaryOperator::PtrMemD:
        case BinaryOperator::PtrMemI:
          assert(false && "Pointer arithmetic not handled here.");
          return UnknownVal();
        case BinaryOperator::LT:
        case BinaryOperator::GT:
        case BinaryOperator::LE:
        case BinaryOperator::GE:
        case BinaryOperator::EQ:
        case BinaryOperator::NE:
          // Negate the comparison and make a value.
          opc = NegateComparison(opc);
          assert(symIntExpr->getType(ValMgr.getContext()) == resultTy);
          return ValMgr.makeNonLoc(symIntExpr->getLHS(), opc,
                                   symIntExpr->getRHS(), resultTy);
        }
      }

      // For now, only handle expressions whose RHS is a constant.
      const nonloc::ConcreteInt *rhsInt = dyn_cast<nonloc::ConcreteInt>(&rhs);
      if (!rhsInt)
        return UnknownVal();

      // If both the LHS and the current expression are additive,
      // fold their constants.
      if (BinaryOperator::isAdditiveOp(op)) {
        BinaryOperator::Opcode lop = symIntExpr->getOpcode();
        if (BinaryOperator::isAdditiveOp(lop)) {
          BasicValueFactory &BVF = ValMgr.getBasicValueFactory();

          // resultTy may not be the best type to convert to, but it's
          // probably the best choice in expressions with mixed type
          // (such as x+1U+2LL). The rules for implicit conversions should
          // choose a reasonable type to preserve the expression, and will
          // at least match how the value is going to be used.
          const llvm::APSInt &first =
            BVF.Convert(resultTy, symIntExpr->getRHS());
          const llvm::APSInt &second =
            BVF.Convert(resultTy, rhsInt->getValue());

          const llvm::APSInt *newRHS;
          if (lop == op)
            newRHS = BVF.EvaluateAPSInt(BinaryOperator::Add, first, second);
          else
            newRHS = BVF.EvaluateAPSInt(BinaryOperator::Sub, first, second);
          return MakeSymIntVal(symIntExpr->getLHS(), lop, *newRHS, resultTy);
        }
      }

      // Otherwise, make a SymExprVal out of the expression.
      return MakeSymIntVal(symIntExpr, op, rhsInt->getValue(), resultTy);
    }
    case nonloc::ConcreteIntKind: {
      const nonloc::ConcreteInt& lhsInt = cast<nonloc::ConcreteInt>(lhs);

      if (isa<nonloc::ConcreteInt>(rhs)) {
        return lhsInt.evalBinOp(ValMgr, op, cast<nonloc::ConcreteInt>(rhs));
      } else {
        const llvm::APSInt& lhsValue = lhsInt.getValue();
        
        // Swap the left and right sides and flip the operator if doing so
        // allows us to better reason about the expression (this is a form
        // of expression canonicalization).
        // While we're at it, catch some special cases for non-commutative ops.
        NonLoc tmp = rhs;
        rhs = lhs;
        lhs = tmp;

        switch (op) {
          case BinaryOperator::LT: op = BinaryOperator::GT; continue;
          case BinaryOperator::GT: op = BinaryOperator::LT; continue;
          case BinaryOperator::LE: op = BinaryOperator::GE; continue;
          case BinaryOperator::GE: op = BinaryOperator::LE; continue;
          case BinaryOperator::EQ:
          case BinaryOperator::NE:
          case BinaryOperator::Add:
          case BinaryOperator::Mul:
          case BinaryOperator::And:
          case BinaryOperator::Xor:
          case BinaryOperator::Or:
            continue;
          case BinaryOperator::Shr:
            if (lhsValue.isAllOnesValue() && lhsValue.isSigned())
              // At this point lhs and rhs have been swapped.
              return rhs;
            // FALL-THROUGH
          case BinaryOperator::Shl:
            if (lhsValue == 0)
              // At this point lhs and rhs have been swapped.
              return rhs;
            return UnknownVal();
          default:
            return UnknownVal();
        }
      }
    }
    case nonloc::SymbolValKind: {
      nonloc::SymbolVal *slhs = cast<nonloc::SymbolVal>(&lhs);
      SymbolRef Sym = slhs->getSymbol();
      
      // Does the symbol simplify to a constant?  If so, "fold" the constant
      // by setting 'lhs' to a ConcreteInt and try again.
      if (Sym->getType(ValMgr.getContext())->isIntegerType())
        if (const llvm::APSInt *Constant = state->getSymVal(Sym)) {
          // The symbol evaluates to a constant. If necessary, promote the
          // folded constant (LHS) to the result type.
          BasicValueFactory &BVF = ValMgr.getBasicValueFactory();
          const llvm::APSInt &lhs_I = BVF.Convert(resultTy, *Constant);
          lhs = nonloc::ConcreteInt(lhs_I);
          
          // Also promote the RHS (if necessary).

          // For shifts, it necessary promote the RHS to the result type.
          if (BinaryOperator::isShiftOp(op))
            continue;
          
          // Other operators: do an implicit conversion.  This shouldn't be
          // necessary once we support truncation/extension of symbolic values.
          if (nonloc::ConcreteInt *rhs_I = dyn_cast<nonloc::ConcreteInt>(&rhs)){
            rhs = nonloc::ConcreteInt(BVF.Convert(resultTy, rhs_I->getValue()));
          }
          
          continue;
        }
      
      if (isa<nonloc::ConcreteInt>(rhs)) {
        return MakeSymIntVal(slhs->getSymbol(), op,
                             cast<nonloc::ConcreteInt>(rhs).getValue(),
                             resultTy);
      }

      return UnknownVal();
    }
    }
  }
}

SVal SimpleSValuator::EvalBinOpLL(BinaryOperator::Opcode op, Loc lhs, Loc rhs,
                                  QualType resultTy) {
  switch (op) {
    default:
      return UnknownVal();
    case BinaryOperator::EQ:
    case BinaryOperator::NE:
      return EvalEquality(ValMgr, lhs, rhs, op == BinaryOperator::EQ, resultTy);
    case BinaryOperator::LT:
    case BinaryOperator::GT:
      // FIXME: Generalize.  For now, just handle the trivial case where
      //  the two locations are identical.
      if (lhs == rhs)
        return ValMgr.makeTruthVal(false, resultTy);
      return UnknownVal();
  }
}

SVal SimpleSValuator::EvalBinOpLN(const GRState *state,
                                  BinaryOperator::Opcode op,
                                  Loc lhs, NonLoc rhs, QualType resultTy) {
  // Special case: 'rhs' is an integer that has the same width as a pointer and
  // we are using the integer location in a comparison.  Normally this cannot be
  // triggered, but transfer functions like those for OSCommpareAndSwapBarrier32
  // can generate comparisons that trigger this code.
  // FIXME: Are all locations guaranteed to have pointer width?
  if (BinaryOperator::isEqualityOp(op)) {
    if (nonloc::ConcreteInt *rhsInt = dyn_cast<nonloc::ConcreteInt>(&rhs)) {
      const llvm::APSInt *x = &rhsInt->getValue();
      ASTContext &ctx = ValMgr.getContext();
      if (ctx.getTypeSize(ctx.VoidPtrTy) == x->getBitWidth()) {
        // Convert the signedness of the integer (if necessary).
        if (x->isSigned())
          x = &ValMgr.getBasicValueFactory().getValue(*x, true);

        return EvalBinOpLL(op, lhs, loc::ConcreteInt(*x), resultTy);
      }
    }
  }

  // Delegate pointer arithmetic to the StoreManager.
  return state->getStateManager().getStoreManager().EvalBinOp(op, lhs,
                                                              rhs, resultTy);
}
