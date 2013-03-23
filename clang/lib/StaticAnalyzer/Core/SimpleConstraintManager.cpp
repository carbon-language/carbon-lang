//== SimpleConstraintManager.cpp --------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SimpleConstraintManager, a class that holds code shared
//  between BasicConstraintManager and RangeConstraintManager.
//
//===----------------------------------------------------------------------===//

#include "SimpleConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/APSIntType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"

namespace clang {

namespace ento {

SimpleConstraintManager::~SimpleConstraintManager() {}

bool SimpleConstraintManager::canReasonAbout(SVal X) const {
  Optional<nonloc::SymbolVal> SymVal = X.getAs<nonloc::SymbolVal>();
  if (SymVal && SymVal->isExpression()) {
    const SymExpr *SE = SymVal->getSymbol();

    if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(SE)) {
      switch (SIE->getOpcode()) {
          // We don't reason yet about bitwise-constraints on symbolic values.
        case BO_And:
        case BO_Or:
        case BO_Xor:
          return false;
        // We don't reason yet about these arithmetic constraints on
        // symbolic values.
        case BO_Mul:
        case BO_Div:
        case BO_Rem:
        case BO_Shl:
        case BO_Shr:
          return false;
        // All other cases.
        default:
          return true;
      }
    }

    if (const SymSymExpr *SSE = dyn_cast<SymSymExpr>(SE)) {
      if (SSE->getOpcode() == BO_EQ || SSE->getOpcode() == BO_NE)
        return true;
    }

    return false;
  }

  return true;
}

ProgramStateRef SimpleConstraintManager::assume(ProgramStateRef state,
                                               DefinedSVal Cond,
                                               bool Assumption) {
  if (Optional<NonLoc> NV = Cond.getAs<NonLoc>())
    return assume(state, *NV, Assumption);
  return assume(state, Cond.castAs<Loc>(), Assumption);
}

ProgramStateRef SimpleConstraintManager::assume(ProgramStateRef state, Loc cond,
                                               bool assumption) {
  state = assumeAux(state, cond, assumption);
  if (NotifyAssumeClients && SU)
    return SU->processAssume(state, cond, assumption);
  return state;
}

ProgramStateRef SimpleConstraintManager::assumeAux(ProgramStateRef state,
                                                  Loc Cond, bool Assumption) {
  switch (Cond.getSubKind()) {
  default:
    assert (false && "'Assume' not implemented for this Loc.");
    return state;

  case loc::MemRegionKind: {
    // FIXME: Should this go into the storemanager?

    const MemRegion *R = Cond.castAs<loc::MemRegionVal>().getRegion();
    const SubRegion *SubR = dyn_cast<SubRegion>(R);

    while (SubR) {
      // FIXME: now we only find the first symbolic region.
      if (const SymbolicRegion *SymR = dyn_cast<SymbolicRegion>(SubR)) {
        const llvm::APSInt &zero = getBasicVals().getZeroWithPtrWidth();
        if (Assumption)
          return assumeSymNE(state, SymR->getSymbol(), zero, zero);
        else
          return assumeSymEQ(state, SymR->getSymbol(), zero, zero);
      }
      SubR = dyn_cast<SubRegion>(SubR->getSuperRegion());
    }

    // FALL-THROUGH.
  }

  case loc::GotoLabelKind:
    return Assumption ? state : NULL;

  case loc::ConcreteIntKind: {
    bool b = Cond.castAs<loc::ConcreteInt>().getValue() != 0;
    bool isFeasible = b ? Assumption : !Assumption;
    return isFeasible ? state : NULL;
  }
  } // end switch
}

ProgramStateRef SimpleConstraintManager::assume(ProgramStateRef state,
                                               NonLoc cond,
                                               bool assumption) {
  state = assumeAux(state, cond, assumption);
  if (NotifyAssumeClients && SU)
    return SU->processAssume(state, cond, assumption);
  return state;
}

static BinaryOperator::Opcode NegateComparison(BinaryOperator::Opcode op) {
  // FIXME: This should probably be part of BinaryOperator, since this isn't
  // the only place it's used. (This code was copied from SimpleSValBuilder.cpp.)
  switch (op) {
  default:
    llvm_unreachable("Invalid opcode.");
  case BO_LT: return BO_GE;
  case BO_GT: return BO_LE;
  case BO_LE: return BO_GT;
  case BO_GE: return BO_LT;
  case BO_EQ: return BO_NE;
  case BO_NE: return BO_EQ;
  }
}


ProgramStateRef
SimpleConstraintManager::assumeAuxForSymbol(ProgramStateRef State,
                                            SymbolRef Sym, bool Assumption) {
  BasicValueFactory &BVF = getBasicVals();
  QualType T = Sym->getType();

  // None of the constraint solvers currently support non-integer types.
  if (!T->isIntegerType())
    return State;

  const llvm::APSInt &zero = BVF.getValue(0, T);
  if (Assumption)
    return assumeSymNE(State, Sym, zero, zero);
  else
    return assumeSymEQ(State, Sym, zero, zero);
}

ProgramStateRef SimpleConstraintManager::assumeAux(ProgramStateRef state,
                                                  NonLoc Cond,
                                                  bool Assumption) {

  // We cannot reason about SymSymExprs, and can only reason about some
  // SymIntExprs.
  if (!canReasonAbout(Cond)) {
    // Just add the constraint to the expression without trying to simplify.
    SymbolRef sym = Cond.getAsSymExpr();
    return assumeAuxForSymbol(state, sym, Assumption);
  }

  switch (Cond.getSubKind()) {
  default:
    llvm_unreachable("'Assume' not implemented for this NonLoc");

  case nonloc::SymbolValKind: {
    nonloc::SymbolVal SV = Cond.castAs<nonloc::SymbolVal>();
    SymbolRef sym = SV.getSymbol();
    assert(sym);

    // Handle SymbolData.
    if (!SV.isExpression()) {
      return assumeAuxForSymbol(state, sym, Assumption);

    // Handle symbolic expression.
    } else if (const SymIntExpr *SE = dyn_cast<SymIntExpr>(sym)) {
      // We can only simplify expressions whose RHS is an integer.

      BinaryOperator::Opcode op = SE->getOpcode();
      if (BinaryOperator::isComparisonOp(op)) {
        if (!Assumption)
          op = NegateComparison(op);

        return assumeSymRel(state, SE->getLHS(), op, SE->getRHS());
      }

    } else if (const SymSymExpr *SSE = dyn_cast<SymSymExpr>(sym)) {
      BinaryOperator::Opcode Op = SSE->getOpcode();

      // Translate "a != b" to "(b - a) != 0".
      // We invert the order of the operands as a heuristic for how loop
      // conditions are usually written ("begin != end") as compared to length
      // calculations ("end - begin"). The more correct thing to do would be to
      // canonicalize "a - b" and "b - a", which would allow us to treat
      // "a != b" and "b != a" the same.
      if (BinaryOperator::isEqualityOp(Op)) {
        SymbolManager &SymMgr = getSymbolManager();

        assert(Loc::isLocType(SSE->getLHS()->getType()));
        assert(Loc::isLocType(SSE->getRHS()->getType()));
        QualType DiffTy = SymMgr.getContext().getPointerDiffType();
        SymbolRef Subtraction = SymMgr.getSymSymExpr(SSE->getRHS(), BO_Sub,
                                                     SSE->getLHS(), DiffTy);

        Assumption ^= (SSE->getOpcode() == BO_EQ);
        return assumeAuxForSymbol(state, Subtraction, Assumption);
      }
    }

    // If we get here, there's nothing else we can do but treat the symbol as
    // opaque.
    return assumeAuxForSymbol(state, sym, Assumption);
  }

  case nonloc::ConcreteIntKind: {
    bool b = Cond.castAs<nonloc::ConcreteInt>().getValue() != 0;
    bool isFeasible = b ? Assumption : !Assumption;
    return isFeasible ? state : NULL;
  }

  case nonloc::LocAsIntegerKind:
    return assumeAux(state, Cond.castAs<nonloc::LocAsInteger>().getLoc(),
                     Assumption);
  } // end switch
}

static void computeAdjustment(SymbolRef &Sym, llvm::APSInt &Adjustment) {
  // Is it a "($sym+constant1)" expression?
  if (const SymIntExpr *SE = dyn_cast<SymIntExpr>(Sym)) {
    BinaryOperator::Opcode Op = SE->getOpcode();
    if (Op == BO_Add || Op == BO_Sub) {
      Sym = SE->getLHS();
      Adjustment = APSIntType(Adjustment).convert(SE->getRHS());

      // Don't forget to negate the adjustment if it's being subtracted.
      // This should happen /after/ promotion, in case the value being
      // subtracted is, say, CHAR_MIN, and the promoted type is 'int'.
      if (Op == BO_Sub)
        Adjustment = -Adjustment;
    }
  }
}

ProgramStateRef SimpleConstraintManager::assumeSymRel(ProgramStateRef state,
                                                     const SymExpr *LHS,
                                                     BinaryOperator::Opcode op,
                                                     const llvm::APSInt& Int) {
  assert(BinaryOperator::isComparisonOp(op) &&
         "Non-comparison ops should be rewritten as comparisons to zero.");

  // Get the type used for calculating wraparound.
  BasicValueFactory &BVF = getBasicVals();
  APSIntType WraparoundType = BVF.getAPSIntType(LHS->getType());

  // We only handle simple comparisons of the form "$sym == constant"
  // or "($sym+constant1) == constant2".
  // The adjustment is "constant1" in the above expression. It's used to
  // "slide" the solution range around for modular arithmetic. For example,
  // x < 4 has the solution [0, 3]. x+2 < 4 has the solution [0-2, 3-2], which
  // in modular arithmetic is [0, 1] U [UINT_MAX-1, UINT_MAX]. It's up to
  // the subclasses of SimpleConstraintManager to handle the adjustment.
  SymbolRef Sym = LHS;
  llvm::APSInt Adjustment = WraparoundType.getZeroValue();
  computeAdjustment(Sym, Adjustment);

  // Convert the right-hand side integer as necessary.
  APSIntType ComparisonType = std::max(WraparoundType, APSIntType(Int));
  llvm::APSInt ConvertedInt = ComparisonType.convert(Int);

  switch (op) {
  default:
    // No logic yet for other operators.  assume the constraint is feasible.
    return state;

  case BO_EQ:
    return assumeSymEQ(state, Sym, ConvertedInt, Adjustment);

  case BO_NE:
    return assumeSymNE(state, Sym, ConvertedInt, Adjustment);

  case BO_GT:
    return assumeSymGT(state, Sym, ConvertedInt, Adjustment);

  case BO_GE:
    return assumeSymGE(state, Sym, ConvertedInt, Adjustment);

  case BO_LT:
    return assumeSymLT(state, Sym, ConvertedInt, Adjustment);

  case BO_LE:
    return assumeSymLE(state, Sym, ConvertedInt, Adjustment);
  } // end switch
}

} // end of namespace ento

} // end of namespace clang
