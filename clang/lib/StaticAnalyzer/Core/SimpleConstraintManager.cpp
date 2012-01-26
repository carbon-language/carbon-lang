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
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"

namespace clang {

namespace ento {

SimpleConstraintManager::~SimpleConstraintManager() {}

bool SimpleConstraintManager::canReasonAbout(SVal X) const {
  nonloc::SymbolVal *SymVal = dyn_cast<nonloc::SymbolVal>(&X);
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

    return false;
  }

  return true;
}

ProgramStateRef SimpleConstraintManager::assume(ProgramStateRef state,
                                               DefinedSVal Cond,
                                               bool Assumption) {
  if (isa<NonLoc>(Cond))
    return assume(state, cast<NonLoc>(Cond), Assumption);
  else
    return assume(state, cast<Loc>(Cond), Assumption);
}

ProgramStateRef SimpleConstraintManager::assume(ProgramStateRef state, Loc cond,
                                               bool assumption) {
  state = assumeAux(state, cond, assumption);
  return SU.processAssume(state, cond, assumption);
}

ProgramStateRef SimpleConstraintManager::assumeAux(ProgramStateRef state,
                                                  Loc Cond, bool Assumption) {

  BasicValueFactory &BasicVals = state->getBasicVals();

  switch (Cond.getSubKind()) {
  default:
    assert (false && "'Assume' not implemented for this Loc.");
    return state;

  case loc::MemRegionKind: {
    // FIXME: Should this go into the storemanager?

    const MemRegion *R = cast<loc::MemRegionVal>(Cond).getRegion();
    const SubRegion *SubR = dyn_cast<SubRegion>(R);

    while (SubR) {
      // FIXME: now we only find the first symbolic region.
      if (const SymbolicRegion *SymR = dyn_cast<SymbolicRegion>(SubR)) {
        const llvm::APSInt &zero = BasicVals.getZeroWithPtrWidth();
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
    bool b = cast<loc::ConcreteInt>(Cond).getValue() != 0;
    bool isFeasible = b ? Assumption : !Assumption;
    return isFeasible ? state : NULL;
  }
  } // end switch
}

ProgramStateRef SimpleConstraintManager::assume(ProgramStateRef state,
                                               NonLoc cond,
                                               bool assumption) {
  state = assumeAux(state, cond, assumption);
  return SU.processAssume(state, cond, assumption);
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


ProgramStateRef SimpleConstraintManager::assumeAuxForSymbol(
                                              ProgramStateRef State,
                                              SymbolRef Sym,
                                              bool Assumption) {
  QualType T =  State->getSymbolManager().getType(Sym);
  const llvm::APSInt &zero = State->getBasicVals().getValue(0, T);
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

  BasicValueFactory &BasicVals = state->getBasicVals();
  SymbolManager &SymMgr = state->getSymbolManager();

  switch (Cond.getSubKind()) {
  default:
    llvm_unreachable("'Assume' not implemented for this NonLoc");

  case nonloc::SymbolValKind: {
    nonloc::SymbolVal& SV = cast<nonloc::SymbolVal>(Cond);
    SymbolRef sym = SV.getSymbol();
    assert(sym);

    // Handle SymbolData.
    if (!SV.isExpression()) {
      return assumeAuxForSymbol(state, sym, Assumption);

    // Handle symbolic expression.
    } else {
      // We can only simplify expressions whose RHS is an integer.
      const SymIntExpr *SE = dyn_cast<SymIntExpr>(sym);
      if (!SE)
        return assumeAuxForSymbol(state, sym, Assumption);

      BinaryOperator::Opcode op = SE->getOpcode();
      // Implicitly compare non-comparison expressions to 0.
      if (!BinaryOperator::isComparisonOp(op)) {
        QualType T = SymMgr.getType(SE);
        const llvm::APSInt &zero = BasicVals.getValue(0, T);
        op = (Assumption ? BO_NE : BO_EQ);
        return assumeSymRel(state, SE, op, zero);
      }
      // From here on out, op is the real comparison we'll be testing.
      if (!Assumption)
        op = NegateComparison(op);

      return assumeSymRel(state, SE->getLHS(), op, SE->getRHS());
    }
  }

  case nonloc::ConcreteIntKind: {
    bool b = cast<nonloc::ConcreteInt>(Cond).getValue() != 0;
    bool isFeasible = b ? Assumption : !Assumption;
    return isFeasible ? state : NULL;
  }

  case nonloc::LocAsIntegerKind:
    return assumeAux(state, cast<nonloc::LocAsInteger>(Cond).getLoc(),
                     Assumption);
  } // end switch
}

static llvm::APSInt computeAdjustment(const SymExpr *LHS,
                                      SymbolRef &Sym) {
  llvm::APSInt DefaultAdjustment;
  DefaultAdjustment = 0;

  // First check if the LHS is a simple symbol reference.
  if (isa<SymbolData>(LHS))
    return DefaultAdjustment;

  // Next, see if it's a "($sym+constant1)" expression.
  const SymIntExpr *SE = dyn_cast<SymIntExpr>(LHS);

  // We cannot simplify "($sym1+$sym2)".
  if (!SE)
    return DefaultAdjustment;

  // Get the constant out of the expression "($sym+constant1)" or
  // "<expr>+constant1".
  Sym = SE->getLHS();
  switch (SE->getOpcode()) {
  case BO_Add:
    return SE->getRHS();
  case BO_Sub:
    return -SE->getRHS();
  default:
    // We cannot simplify non-additive operators.
    return DefaultAdjustment;
  }
}

ProgramStateRef SimpleConstraintManager::assumeSymRel(ProgramStateRef state,
                                                     const SymExpr *LHS,
                                                     BinaryOperator::Opcode op,
                                                     const llvm::APSInt& Int) {
  assert(BinaryOperator::isComparisonOp(op) &&
         "Non-comparison ops should be rewritten as comparisons to zero.");

  // We only handle simple comparisons of the form "$sym == constant"
  // or "($sym+constant1) == constant2".
  // The adjustment is "constant1" in the above expression. It's used to
  // "slide" the solution range around for modular arithmetic. For example,
  // x < 4 has the solution [0, 3]. x+2 < 4 has the solution [0-2, 3-2], which
  // in modular arithmetic is [0, 1] U [UINT_MAX-1, UINT_MAX]. It's up to
  // the subclasses of SimpleConstraintManager to handle the adjustment.
  SymbolRef Sym = LHS;
  llvm::APSInt Adjustment = computeAdjustment(LHS, Sym);

  // FIXME: This next section is a hack. It silently converts the integers to
  // be of the same type as the symbol, which is not always correct. Really the
  // comparisons should be performed using the Int's type, then mapped back to
  // the symbol's range of values.
  ProgramStateManager &StateMgr = state->getStateManager();
  ASTContext &Ctx = StateMgr.getContext();

  QualType T = Sym->getType(Ctx);
  assert(T->isIntegerType() || Loc::isLocType(T));
  unsigned bitwidth = Ctx.getTypeSize(T);
  bool isSymUnsigned 
    = T->isUnsignedIntegerOrEnumerationType() || Loc::isLocType(T);

  // Convert the adjustment.
  Adjustment.setIsUnsigned(isSymUnsigned);
  Adjustment = Adjustment.extOrTrunc(bitwidth);

  // Convert the right-hand side integer.
  llvm::APSInt ConvertedInt(Int, isSymUnsigned);
  ConvertedInt = ConvertedInt.extOrTrunc(bitwidth);

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
