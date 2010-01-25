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
#include "clang/Checker/PathSensitive/GRExprEngine.h"
#include "clang/Checker/PathSensitive/GRState.h"
#include "clang/Checker/PathSensitive/Checker.h"

namespace clang {

SimpleConstraintManager::~SimpleConstraintManager() {}

bool SimpleConstraintManager::canReasonAbout(SVal X) const {
  if (nonloc::SymExprVal *SymVal = dyn_cast<nonloc::SymExprVal>(&X)) {
    const SymExpr *SE = SymVal->getSymbolicExpression();

    if (isa<SymbolData>(SE))
      return true;

    if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(SE)) {
      switch (SIE->getOpcode()) {
          // We don't reason yet about bitwise-constraints on symbolic values.
        case BinaryOperator::And:
        case BinaryOperator::Or:
        case BinaryOperator::Xor:
          return false;
        // We don't reason yet about arithmetic constraints on symbolic values.
        case BinaryOperator::Mul:
        case BinaryOperator::Div:
        case BinaryOperator::Rem:
        case BinaryOperator::Add:
        case BinaryOperator::Sub:
        case BinaryOperator::Shl:
        case BinaryOperator::Shr:
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

const GRState *SimpleConstraintManager::Assume(const GRState *state,
                                               DefinedSVal Cond,
                                               bool Assumption) {
  if (isa<NonLoc>(Cond))
    return Assume(state, cast<NonLoc>(Cond), Assumption);
  else
    return Assume(state, cast<Loc>(Cond), Assumption);
}

const GRState *SimpleConstraintManager::Assume(const GRState *state, Loc cond,
                                               bool assumption) {
  state = AssumeAux(state, cond, assumption);
  return SU.ProcessAssume(state, cond, assumption);
}

const GRState *SimpleConstraintManager::AssumeAux(const GRState *state,
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
        if (Assumption)
          return AssumeSymNE(state, SymR->getSymbol(),
                             BasicVals.getZeroWithPtrWidth());
        else
          return AssumeSymEQ(state, SymR->getSymbol(),
                             BasicVals.getZeroWithPtrWidth());
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

const GRState *SimpleConstraintManager::Assume(const GRState *state,
                                               NonLoc cond,
                                               bool assumption) {
  state = AssumeAux(state, cond, assumption);
  return SU.ProcessAssume(state, cond, assumption);
}

const GRState *SimpleConstraintManager::AssumeAux(const GRState *state,
                                                  NonLoc Cond,
                                                  bool Assumption) {

  // We cannot reason about SymIntExpr and SymSymExpr.
  if (!canReasonAbout(Cond)) {
    // Just return the current state indicating that the path is feasible.
    // This may be an over-approximation of what is possible.
    return state;
  }

  BasicValueFactory &BasicVals = state->getBasicVals();
  SymbolManager &SymMgr = state->getSymbolManager();

  switch (Cond.getSubKind()) {
  default:
    assert(false && "'Assume' not implemented for this NonLoc");

  case nonloc::SymbolValKind: {
    nonloc::SymbolVal& SV = cast<nonloc::SymbolVal>(Cond);
    SymbolRef sym = SV.getSymbol();
    QualType T =  SymMgr.getType(sym);
    const llvm::APSInt &zero = BasicVals.getValue(0, T);

    return Assumption ? AssumeSymNE(state, sym, zero)
                      : AssumeSymEQ(state, sym, zero);
  }

  case nonloc::SymExprValKind: {
    nonloc::SymExprVal V = cast<nonloc::SymExprVal>(Cond);
    if (const SymIntExpr *SE = dyn_cast<SymIntExpr>(V.getSymbolicExpression())){
      // FIXME: This is a hack.  It silently converts the RHS integer to be
      // of the same type as on the left side.  This should be removed once
      // we support truncation/extension of symbolic values.      
      GRStateManager &StateMgr = state->getStateManager();
      ASTContext &Ctx = StateMgr.getContext();
      QualType LHSType = SE->getLHS()->getType(Ctx);
      BasicValueFactory &BasicVals = StateMgr.getBasicVals();
      const llvm::APSInt &RHS = BasicVals.Convert(LHSType, SE->getRHS());
      SymIntExpr SENew(SE->getLHS(), SE->getOpcode(), RHS, SE->getType(Ctx));

      return AssumeSymInt(state, Assumption, &SENew);
    }

    // For all other symbolic expressions, over-approximate and consider
    // the constraint feasible.
    return state;
  }

  case nonloc::ConcreteIntKind: {
    bool b = cast<nonloc::ConcreteInt>(Cond).getValue() != 0;
    bool isFeasible = b ? Assumption : !Assumption;
    return isFeasible ? state : NULL;
  }

  case nonloc::LocAsIntegerKind:
    return AssumeAux(state, cast<nonloc::LocAsInteger>(Cond).getLoc(),
                     Assumption);
  } // end switch
}

const GRState *SimpleConstraintManager::AssumeSymInt(const GRState *state,
                                                     bool Assumption,
                                                     const SymIntExpr *SE) {


  // Here we assume that LHS is a symbol.  This is consistent with the
  // rest of the constraint manager logic.
  SymbolRef Sym = cast<SymbolData>(SE->getLHS());
  const llvm::APSInt &Int = SE->getRHS();

  switch (SE->getOpcode()) {
  default:
    // No logic yet for other operators.  Assume the constraint is feasible.
    return state;

  case BinaryOperator::EQ:
    return Assumption ? AssumeSymEQ(state, Sym, Int)
                      : AssumeSymNE(state, Sym, Int);

  case BinaryOperator::NE:
    return Assumption ? AssumeSymNE(state, Sym, Int)
                      : AssumeSymEQ(state, Sym, Int);
  case BinaryOperator::GT:
    return Assumption ? AssumeSymGT(state, Sym, Int)
                      : AssumeSymLE(state, Sym, Int);

  case BinaryOperator::GE:
    return Assumption ? AssumeSymGE(state, Sym, Int)
                      : AssumeSymLT(state, Sym, Int);

  case BinaryOperator::LT:
    return Assumption ? AssumeSymLT(state, Sym, Int)
                      : AssumeSymGE(state, Sym, Int);

  case BinaryOperator::LE:
    return Assumption ? AssumeSymLE(state, Sym, Int)
                      : AssumeSymGT(state, Sym, Int);
  } // end switch
}

const GRState *SimpleConstraintManager::AssumeInBound(const GRState *state,
                                                      DefinedSVal Idx,
                                                      DefinedSVal UpperBound,
                                                      bool Assumption) {

  // Only support ConcreteInt for now.
  if (!(isa<nonloc::ConcreteInt>(Idx) && isa<nonloc::ConcreteInt>(UpperBound)))
    return state;

  const llvm::APSInt& Zero = state->getBasicVals().getZeroWithPtrWidth(false);
  llvm::APSInt IdxV = cast<nonloc::ConcreteInt>(Idx).getValue();
  // IdxV might be too narrow.
  if (IdxV.getBitWidth() < Zero.getBitWidth())
    IdxV.extend(Zero.getBitWidth());
  // UBV might be too narrow, too.
  llvm::APSInt UBV = cast<nonloc::ConcreteInt>(UpperBound).getValue();
  if (UBV.getBitWidth() < Zero.getBitWidth())
    UBV.extend(Zero.getBitWidth());

  bool InBound = (Zero <= IdxV) && (IdxV < UBV);
  bool isFeasible = Assumption ? InBound : !InBound;
  return isFeasible ? state : NULL;
}

}  // end of namespace clang
