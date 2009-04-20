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
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/PathSensitive/GRState.h"

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
  
const GRState*
SimpleConstraintManager::Assume(const GRState* St, SVal Cond, bool Assumption,
                                bool& isFeasible) {
  if (Cond.isUnknown()) {
    isFeasible = true;
    return St;
  }

  if (isa<NonLoc>(Cond))
    return Assume(St, cast<NonLoc>(Cond), Assumption, isFeasible);
  else
    return Assume(St, cast<Loc>(Cond), Assumption, isFeasible);
}

const GRState*
SimpleConstraintManager::Assume(const GRState* St, Loc Cond, bool Assumption,
                                bool& isFeasible) {
  St = AssumeAux(St, Cond, Assumption, isFeasible);
  
  if (!isFeasible)
    return St;
  
  // EvalAssume is used to call into the GRTransferFunction object to perform
  // any checker-specific update of the state based on this assumption being
  // true or false.
  return StateMgr.getTransferFuncs().EvalAssume(StateMgr, St, Cond, Assumption,
                                                isFeasible);
}

const GRState*
SimpleConstraintManager::AssumeAux(const GRState* St, Loc Cond, bool Assumption,
                                   bool& isFeasible) {
  BasicValueFactory& BasicVals = StateMgr.getBasicVals();

  switch (Cond.getSubKind()) {
  default:
    assert (false && "'Assume' not implemented for this Loc.");
    return St;

  case loc::MemRegionKind: {
    // FIXME: Should this go into the storemanager?
    
    const MemRegion* R = cast<loc::MemRegionVal>(Cond).getRegion();
    const SubRegion* SubR = dyn_cast<SubRegion>(R);

    while (SubR) {
      // FIXME: now we only find the first symbolic region.
      if (const SymbolicRegion* SymR = dyn_cast<SymbolicRegion>(SubR)) {
        if (Assumption)
          return AssumeSymNE(St, SymR->getSymbol(),
                             BasicVals.getZeroWithPtrWidth(), isFeasible);
        else
          return AssumeSymEQ(St, SymR->getSymbol(),
                             BasicVals.getZeroWithPtrWidth(), isFeasible);
      }
      SubR = dyn_cast<SubRegion>(SubR->getSuperRegion());
    }
    
    // FALL-THROUGH.
  }
      
  case loc::GotoLabelKind:
    isFeasible = Assumption;
    return St;

  case loc::ConcreteIntKind: {
    bool b = cast<loc::ConcreteInt>(Cond).getValue() != 0;
    isFeasible = b ? Assumption : !Assumption;
    return St;
  }
  } // end switch
}

const GRState*
SimpleConstraintManager::Assume(const GRState* St, NonLoc Cond, bool Assumption,
                               bool& isFeasible) {
  St = AssumeAux(St, Cond, Assumption, isFeasible);
  
  if (!isFeasible)
    return St;
  
  // EvalAssume is used to call into the GRTransferFunction object to perform
  // any checker-specific update of the state based on this assumption being
  // true or false.
  return StateMgr.getTransferFuncs().EvalAssume(StateMgr, St, Cond, Assumption,
                                                  isFeasible);
}

const GRState*
SimpleConstraintManager::AssumeAux(const GRState* St,NonLoc Cond,
                                   bool Assumption, bool& isFeasible) {
  // We cannot reason about SymIntExpr and SymSymExpr.
  if (!canReasonAbout(Cond)) {
    isFeasible = true;
    return St;
  }  

  BasicValueFactory& BasicVals = StateMgr.getBasicVals();
  SymbolManager& SymMgr = StateMgr.getSymbolManager();

  switch (Cond.getSubKind()) {
  default:
    assert(false && "'Assume' not implemented for this NonLoc");

  case nonloc::SymbolValKind: {
    nonloc::SymbolVal& SV = cast<nonloc::SymbolVal>(Cond);
    SymbolRef sym = SV.getSymbol();
    QualType T =  SymMgr.getType(sym);
    
    if (Assumption)
      return AssumeSymNE(St, sym, BasicVals.getValue(0, T), isFeasible);
    else
      return AssumeSymEQ(St, sym, BasicVals.getValue(0, T), isFeasible);
  }

  case nonloc::SymExprValKind: {
    nonloc::SymExprVal V = cast<nonloc::SymExprVal>(Cond);
    if (const SymIntExpr *SE = dyn_cast<SymIntExpr>(V.getSymbolicExpression()))
      return AssumeSymInt(St, Assumption, SE, isFeasible);
    
    isFeasible = true;
    return St;
  }

  case nonloc::ConcreteIntKind: {
    bool b = cast<nonloc::ConcreteInt>(Cond).getValue() != 0;
    isFeasible = b ? Assumption : !Assumption;
    return St;
  }

  case nonloc::LocAsIntegerKind:
    return AssumeAux(St, cast<nonloc::LocAsInteger>(Cond).getLoc(),
                     Assumption, isFeasible);
  } // end switch
}

const GRState*
SimpleConstraintManager::AssumeSymInt(const GRState* St, bool Assumption,
                                      const SymIntExpr *SE, bool& isFeasible) {


  // Here we assume that LHS is a symbol.  This is consistent with the
  // rest of the constraint manager logic.
  SymbolRef Sym = cast<SymbolData>(SE->getLHS());
  const llvm::APSInt &Int = SE->getRHS();
  
  switch (SE->getOpcode()) {
  default:
    // No logic yet for other operators.
    isFeasible = true;
    return St;

  case BinaryOperator::EQ:
    return Assumption ? AssumeSymEQ(St, Sym, Int, isFeasible)
                      : AssumeSymNE(St, Sym, Int, isFeasible);

  case BinaryOperator::NE:
    return Assumption ? AssumeSymNE(St, Sym, Int, isFeasible)
                      : AssumeSymEQ(St, Sym, Int, isFeasible);

  case BinaryOperator::GT:
    return Assumption ? AssumeSymGT(St, Sym, Int, isFeasible)
                      : AssumeSymLE(St, Sym, Int, isFeasible);

  case BinaryOperator::GE:
    return Assumption ? AssumeSymGE(St, Sym, Int, isFeasible)
                      : AssumeSymLT(St, Sym, Int, isFeasible);

  case BinaryOperator::LT:
    return Assumption ? AssumeSymLT(St, Sym, Int, isFeasible)
                      : AssumeSymGE(St, Sym, Int, isFeasible);
      
  case BinaryOperator::LE:
      return Assumption ? AssumeSymLE(St, Sym, Int, isFeasible)
                        : AssumeSymGT(St, Sym, Int, isFeasible);
  } // end switch
}

const GRState* 
SimpleConstraintManager::AssumeInBound(const GRState* St, SVal Idx, 
                                       SVal UpperBound, bool Assumption, 
                                       bool& isFeasible) {
  // Only support ConcreteInt for now.
  if (!(isa<nonloc::ConcreteInt>(Idx) && isa<nonloc::ConcreteInt>(UpperBound))){
    isFeasible = true;
    return St;
  }

  const llvm::APSInt& Zero = getBasicVals().getZeroWithPtrWidth(false);
  llvm::APSInt IdxV = cast<nonloc::ConcreteInt>(Idx).getValue();
  // IdxV might be too narrow.
  if (IdxV.getBitWidth() < Zero.getBitWidth())
    IdxV.extend(Zero.getBitWidth());
  // UBV might be too narrow, too.
  llvm::APSInt UBV = cast<nonloc::ConcreteInt>(UpperBound).getValue();
  if (UBV.getBitWidth() < Zero.getBitWidth())
    UBV.extend(Zero.getBitWidth());

  bool InBound = (Zero <= IdxV) && (IdxV < UBV);

  isFeasible = Assumption ? InBound : !InBound;

  return St;
}

}  // end of namespace clang
