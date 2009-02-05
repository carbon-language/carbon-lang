//== BasicConstraintManager.cpp - Manage basic constraints.------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BasicConstraintManager, a class that tracks simple 
//  equality and inequality constraints on symbolic values of GRState.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/ConstraintManager.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathSensitive/GRStateTrait.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;


namespace { class VISIBILITY_HIDDEN ConstNotEq {}; }
namespace { class VISIBILITY_HIDDEN ConstEq {}; }

typedef llvm::ImmutableMap<SymbolRef,GRState::IntSetTy> ConstNotEqTy;
typedef llvm::ImmutableMap<SymbolRef,const llvm::APSInt*> ConstEqTy;
  
static int ConstEqIndex = 0;
static int ConstNotEqIndex = 0;

namespace clang {
template<>
struct GRStateTrait<ConstNotEq> : public GRStatePartialTrait<ConstNotEqTy> {
  static inline void* GDMIndex() { return &ConstNotEqIndex; }  
};

template<>
struct GRStateTrait<ConstEq> : public GRStatePartialTrait<ConstEqTy> {
  static inline void* GDMIndex() { return &ConstEqIndex; }  
};
}  
  
namespace {
// BasicConstraintManager only tracks equality and inequality constraints of
// constants and integer variables.
class VISIBILITY_HIDDEN BasicConstraintManager : public ConstraintManager {
  GRStateManager& StateMgr;
  GRState::IntSetTy::Factory ISetFactory;
public:
  BasicConstraintManager(GRStateManager& statemgr) 
    : StateMgr(statemgr), ISetFactory(statemgr.getAllocator()) {}

  virtual const GRState* Assume(const GRState* St, SVal Cond,
                                bool Assumption, bool& isFeasible);

  const GRState* Assume(const GRState* St, Loc Cond, bool Assumption,
                        bool& isFeasible);

  const GRState* AssumeAux(const GRState* St, Loc Cond,bool Assumption,
                           bool& isFeasible);

  const GRState* Assume(const GRState* St, NonLoc Cond, bool Assumption,
                        bool& isFeasible);

  const GRState* AssumeAux(const GRState* St, NonLoc Cond, bool Assumption,
                           bool& isFeasible);

  const GRState* AssumeSymInt(const GRState* St, bool Assumption,
                              const SymIntConstraint& C, bool& isFeasible);

  const GRState* AssumeSymNE(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymEQ(const GRState* St, SymbolRef sym,
                                const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymLT(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymGT(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymGE(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymLE(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeInBound(const GRState* St, SVal Idx, SVal UpperBound,
                               bool Assumption, bool& isFeasible);

  const GRState* AddEQ(const GRState* St, SymbolRef sym, const llvm::APSInt& V);

  const GRState* AddNE(const GRState* St, SymbolRef sym, const llvm::APSInt& V);

  const llvm::APSInt* getSymVal(const GRState* St, SymbolRef sym);
  bool isNotEqual(const GRState* St, SymbolRef sym, const llvm::APSInt& V) const;
  bool isEqual(const GRState* St, SymbolRef sym, const llvm::APSInt& V) const;

  const GRState* RemoveDeadBindings(const GRState* St, SymbolReaper& SymReaper);


  void print(const GRState* St, std::ostream& Out, 
             const char* nl, const char *sep);

private:
  BasicValueFactory& getBasicVals() { return StateMgr.getBasicVals(); }
};

} // end anonymous namespace

ConstraintManager* clang::CreateBasicConstraintManager(GRStateManager& StateMgr)
{
  return new BasicConstraintManager(StateMgr);
}

const GRState* BasicConstraintManager::Assume(const GRState* St, SVal Cond,
                                            bool Assumption, bool& isFeasible) {
  if (Cond.isUnknown()) {
    isFeasible = true;
    return St;
  }

  if (isa<NonLoc>(Cond))
    return Assume(St, cast<NonLoc>(Cond), Assumption, isFeasible);
  else
    return Assume(St, cast<Loc>(Cond), Assumption, isFeasible);
}

const GRState* BasicConstraintManager::Assume(const GRState* St, Loc Cond,
                                            bool Assumption, bool& isFeasible) {
  St = AssumeAux(St, Cond, Assumption, isFeasible);
  
  if (!isFeasible)
    return St;
  
  // EvalAssume is used to call into the GRTransferFunction object to perform
  // any checker-specific update of the state based on this assumption being
  // true or false.
  return StateMgr.getTransferFuncs().EvalAssume(StateMgr, St, Cond, Assumption,
                                                isFeasible);
}

const GRState* BasicConstraintManager::AssumeAux(const GRState* St, Loc Cond,
                                            bool Assumption, bool& isFeasible) {
  BasicValueFactory& BasicVals = StateMgr.getBasicVals();

  switch (Cond.getSubKind()) {
  default:
    assert (false && "'Assume' not implemented for this Loc.");
    return St;

  case loc::SymbolValKind:
    if (Assumption)
      return AssumeSymNE(St, cast<loc::SymbolVal>(Cond).getSymbol(),
                         BasicVals.getZeroWithPtrWidth(), isFeasible);
    else
      return AssumeSymEQ(St, cast<loc::SymbolVal>(Cond).getSymbol(),
                         BasicVals.getZeroWithPtrWidth(), isFeasible);

  case loc::MemRegionKind: {
    // FIXME: Should this go into the storemanager?
    
    const MemRegion* R = cast<loc::MemRegionVal>(Cond).getRegion();
    const SubRegion* SubR = dyn_cast<SubRegion>(R);

    while (SubR) {
      // FIXME: now we only find the first symbolic region.
      if (const SymbolicRegion* SymR = dyn_cast<SymbolicRegion>(SubR))
        return AssumeAux(St, loc::SymbolVal(SymR->getSymbol()), Assumption,
                                            isFeasible);
      SubR = dyn_cast<SubRegion>(SubR->getSuperRegion());
    }
    
    // FALL-THROUGH.
  }
      
  case loc::FuncValKind:
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
BasicConstraintManager::Assume(const GRState* St, NonLoc Cond, bool Assumption,
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
BasicConstraintManager::AssumeAux(const GRState* St,NonLoc Cond,
                                  bool Assumption, bool& isFeasible) {
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

  case nonloc::SymIntConstraintValKind:
    return
      AssumeSymInt(St, Assumption,
                   cast<nonloc::SymIntConstraintVal>(Cond).getConstraint(),
                   isFeasible);

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
BasicConstraintManager::AssumeSymInt(const GRState* St, bool Assumption,
                                  const SymIntConstraint& C, bool& isFeasible) {

  switch (C.getOpcode()) {
  default:
    // No logic yet for other operators.
    isFeasible = true;
    return St;

  case BinaryOperator::EQ:
    if (Assumption)
      return AssumeSymEQ(St, C.getSymbol(), C.getInt(), isFeasible);
    else
      return AssumeSymNE(St, C.getSymbol(), C.getInt(), isFeasible);

  case BinaryOperator::NE:
    if (Assumption)
      return AssumeSymNE(St, C.getSymbol(), C.getInt(), isFeasible);
    else
      return AssumeSymEQ(St, C.getSymbol(), C.getInt(), isFeasible);

  case BinaryOperator::GT:
    if (Assumption)
      return AssumeSymGT(St, C.getSymbol(), C.getInt(), isFeasible);
    else
      return AssumeSymLE(St, C.getSymbol(), C.getInt(), isFeasible);

  case BinaryOperator::GE:
    if (Assumption)
      return AssumeSymGE(St, C.getSymbol(), C.getInt(), isFeasible);
    else
      return AssumeSymLT(St, C.getSymbol(), C.getInt(), isFeasible);

  case BinaryOperator::LT:
    if (Assumption)
      return AssumeSymLT(St, C.getSymbol(), C.getInt(), isFeasible);
    else
      return AssumeSymGE(St, C.getSymbol(), C.getInt(), isFeasible);
      
  case BinaryOperator::LE:
    if (Assumption)
      return AssumeSymLE(St, C.getSymbol(), C.getInt(), isFeasible);
    else
      return AssumeSymGT(St, C.getSymbol(), C.getInt(), isFeasible);
  } // end switch
}

const GRState*
BasicConstraintManager::AssumeSymNE(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible) {
  // First, determine if sym == X, where X != V.
  if (const llvm::APSInt* X = getSymVal(St, sym)) {
    isFeasible = (*X != V);
    return St;
  }

  // Second, determine if sym != V.
  if (isNotEqual(St, sym, V)) {
    isFeasible = true;
    return St;
  }

  // If we reach here, sym is not a constant and we don't know if it is != V.
  // Make that assumption.
  isFeasible = true;
  return AddNE(St, sym, V);
}

const GRState*
BasicConstraintManager::AssumeSymEQ(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible) {
  // First, determine if sym == X, where X != V.
  if (const llvm::APSInt* X = getSymVal(St, sym)) {
    isFeasible = *X == V;
    return St;
  }

  // Second, determine if sym != V.
  if (isNotEqual(St, sym, V)) {
    isFeasible = false;
    return St;
  }

  // If we reach here, sym is not a constant and we don't know if it is == V.
  // Make that assumption.

  isFeasible = true;
  return AddEQ(St, sym, V);
}

// These logic will be handled in another ConstraintManager.
const GRState*
BasicConstraintManager::AssumeSymLT(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible) {
  
  // Is 'V' the smallest possible value?
  if (V == llvm::APSInt::getMinValue(V.getBitWidth(), V.isUnsigned())) {
    // sym cannot be any value less than 'V'.  This path is infeasible.
    isFeasible = false;
    return St;
  }

  // FIXME: For now have assuming x < y be the same as assuming sym != V;
  return AssumeSymNE(St, sym, V, isFeasible);
}

const GRState*
BasicConstraintManager::AssumeSymGT(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  // Is 'V' the largest possible value?
  if (V == llvm::APSInt::getMaxValue(V.getBitWidth(), V.isUnsigned())) {
    // sym cannot be any value greater than 'V'.  This path is infeasible.
    isFeasible = false;
    return St;
  }

  // FIXME: For now have assuming x > y be the same as assuming sym != V;
  return AssumeSymNE(St, sym, V, isFeasible);
}

const GRState*
BasicConstraintManager::AssumeSymGE(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  // Reject a path if the value of sym is a constant X and !(X >= V).
  if (const llvm::APSInt* X = getSymVal(St, sym)) {
    isFeasible = *X >= V;
    return St;
  }
  
  // Sym is not a constant, but it is worth looking to see if V is the
  // maximum integer value.
  if (V == llvm::APSInt::getMaxValue(V.getBitWidth(), V.isUnsigned())) {
    // If we know that sym != V, then this condition is infeasible since
    // there is no other value greater than V.    
    isFeasible = !isNotEqual(St, sym, V);
    
    // If the path is still feasible then as a consequence we know that
    // 'sym == V' because we cannot have 'sym > V' (no larger values).
    // Add this constraint.
    if (isFeasible)
      return AddEQ(St, sym, V);
  }
  else
    isFeasible = true;

  return St;
}

const GRState*
BasicConstraintManager::AssumeSymLE(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  // Reject a path if the value of sym is a constant X and !(X <= V).
  if (const llvm::APSInt* X = getSymVal(St, sym)) {
    isFeasible = *X <= V;
    return St;
  }
  
  // Sym is not a constant, but it is worth looking to see if V is the
  // minimum integer value.
  if (V == llvm::APSInt::getMinValue(V.getBitWidth(), V.isUnsigned())) {
    // If we know that sym != V, then this condition is infeasible since
    // there is no other value less than V.    
    isFeasible = !isNotEqual(St, sym, V);
    
    // If the path is still feasible then as a consequence we know that
    // 'sym == V' because we cannot have 'sym < V' (no smaller values).
    // Add this constraint.
    if (isFeasible)
      return AddEQ(St, sym, V);
  }
  else
    isFeasible = true;
    
  return St;
}

const GRState* 
BasicConstraintManager::AssumeInBound(const GRState* St, SVal Idx, 
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


const GRState* BasicConstraintManager::AddEQ(const GRState* St, SymbolRef sym,
                                             const llvm::APSInt& V) {
  // Create a new state with the old binding replaced.
  GRStateRef state(St, StateMgr);
  return state.set<ConstEq>(sym, &V);
}

const GRState* BasicConstraintManager::AddNE(const GRState* St, SymbolRef sym,
                                             const llvm::APSInt& V) {

  GRStateRef state(St, StateMgr);

  // First, retrieve the NE-set associated with the given symbol.
  ConstNotEqTy::data_type* T = state.get<ConstNotEq>(sym);
  GRState::IntSetTy S = T ? *T : ISetFactory.GetEmptySet();

  
  // Now add V to the NE set.
  S = ISetFactory.Add(S, &V);
  
  // Create a new state with the old binding replaced.
  return state.set<ConstNotEq>(sym, S);
}

const llvm::APSInt* BasicConstraintManager::getSymVal(const GRState* St,
                                                      SymbolRef sym) {
  const ConstEqTy::data_type* T = St->get<ConstEq>(sym);
  return T ? *T : NULL;  
}

bool BasicConstraintManager::isNotEqual(const GRState* St, SymbolRef sym, 
                                        const llvm::APSInt& V) const {

  // Retrieve the NE-set associated with the given symbol.
  const ConstNotEqTy::data_type* T = St->get<ConstNotEq>(sym);

  // See if V is present in the NE-set.
  return T ? T->contains(&V) : false;
}

bool BasicConstraintManager::isEqual(const GRState* St, SymbolRef sym,
                                     const llvm::APSInt& V) const {
  // Retrieve the EQ-set associated with the given symbol.
  const ConstEqTy::data_type* T = St->get<ConstEq>(sym);
  // See if V is present in the EQ-set.
  return T ? **T == V : false;
}

/// Scan all symbols referenced by the constraints. If the symbol is not alive
/// as marked in LSymbols, mark it as dead in DSymbols.
const GRState*
BasicConstraintManager::RemoveDeadBindings(const GRState* St,
                                           SymbolReaper& SymReaper) {

  GRStateRef state(St, StateMgr);
  ConstEqTy CE = state.get<ConstEq>();
  ConstEqTy::Factory& CEFactory = state.get_context<ConstEq>();

  for (ConstEqTy::iterator I = CE.begin(), E = CE.end(); I!=E; ++I) {
    SymbolRef sym = I.getKey();
    if (SymReaper.maybeDead(sym)) CE = CEFactory.Remove(CE, sym);
  }
  state = state.set<ConstEq>(CE);

  ConstNotEqTy CNE = state.get<ConstNotEq>();
  ConstNotEqTy::Factory& CNEFactory = state.get_context<ConstNotEq>();

  for (ConstNotEqTy::iterator I = CNE.begin(), E = CNE.end(); I != E; ++I) {
    SymbolRef sym = I.getKey();    
    if (SymReaper.maybeDead(sym)) CNE = CNEFactory.Remove(CNE, sym);
  }
  
  return state.set<ConstNotEq>(CNE);
}

void BasicConstraintManager::print(const GRState* St, std::ostream& Out, 
                                   const char* nl, const char *sep) {
  // Print equality constraints.

  ConstEqTy CE = St->get<ConstEq>();

  if (!CE.isEmpty()) {
    Out << nl << sep << "'==' constraints:";

    for (ConstEqTy::iterator I = CE.begin(), E = CE.end(); I!=E; ++I) {
      Out << nl << " $" << I.getKey();
      llvm::raw_os_ostream OS(Out);
      OS << " : "   << *I.getData();
    }
  }

  // Print != constraints.
  
  ConstNotEqTy CNE = St->get<ConstNotEq>();
  
  if (!CNE.isEmpty()) {
    Out << nl << sep << "'!=' constraints:";
  
    for (ConstNotEqTy::iterator I = CNE.begin(), EI = CNE.end(); I!=EI; ++I) {
      Out << nl << " $" << I.getKey() << " : ";
      bool isFirst = true;
    
      GRState::IntSetTy::iterator J = I.getData().begin(), 
                                  EJ = I.getData().end();      
      
      for ( ; J != EJ; ++J) {        
        if (isFirst) isFirst = false;
        else Out << ", ";
      
        Out << (*J)->getSExtValue(); // Hack: should print to raw_ostream.
      }
    }
  }
}
