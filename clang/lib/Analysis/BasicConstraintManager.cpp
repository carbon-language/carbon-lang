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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {

typedef llvm::ImmutableMap<SymbolID,GRState::IntSetTy> ConstNotEqTy;
typedef llvm::ImmutableMap<SymbolID,const llvm::APSInt*> ConstEqTy;

// BasicConstraintManager only tracks equality and inequality constraints of
// constants and integer variables.
class VISIBILITY_HIDDEN BasicConstraintManager : public ConstraintManager {
  GRStateManager& StateMgr;

public:
  BasicConstraintManager(GRStateManager& statemgr) : StateMgr(statemgr) {}

  virtual const GRState* Assume(const GRState* St, RVal Cond,
                                bool Assumption, bool& isFeasible);

  const GRState* Assume(const GRState* St, LVal Cond, bool Assumption,
                        bool& isFeasible);

  const GRState* AssumeAux(const GRState* St, LVal Cond,bool Assumption,
                           bool& isFeasible);

  const GRState* Assume(const GRState* St, NonLVal Cond, bool Assumption,
                        bool& isFeasible);

  const GRState* AssumeAux(const GRState* St, NonLVal Cond, bool Assumption,
                           bool& isFeasible);

  const GRState* AssumeSymInt(const GRState* St, bool Assumption,
                              const SymIntConstraint& C, bool& isFeasible);

  const GRState* AssumeSymNE(const GRState* St, SymbolID sym,
                                const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymEQ(const GRState* St, SymbolID sym,
                                const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymLT(const GRState* St, SymbolID sym,
                                    const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymGT(const GRState* St, SymbolID sym,
                             const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymGE(const GRState* St, SymbolID sym,
                             const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymLE(const GRState* St, SymbolID sym,
                             const llvm::APSInt& V, bool& isFeasible);

  const GRState* AddEQ(const GRState* St, SymbolID sym, const llvm::APSInt& V);

  const GRState* AddNE(const GRState* St, SymbolID sym, const llvm::APSInt& V);

  const llvm::APSInt* getSymVal(const GRState* St, SymbolID sym);
  bool isNotEqual(const GRState* St, SymbolID sym, const llvm::APSInt& V) const;
  bool isEqual(const GRState* St, SymbolID sym, const llvm::APSInt& V) const;

  const GRState* RemoveDeadBindings(const GRState* St,
                                    StoreManager::LiveSymbolsTy& LSymbols,
                                    StoreManager::DeadSymbolsTy& DSymbols);

  void print(const GRState* St, std::ostream& Out, 
             const char* nl, const char *sep);
};

} // end anonymous namespace

ConstraintManager* clang::CreateBasicConstraintManager(GRStateManager& StateMgr)
{
  return new BasicConstraintManager(StateMgr);
}

const GRState* BasicConstraintManager::Assume(const GRState* St, RVal Cond,
                                            bool Assumption, bool& isFeasible) {
  if (Cond.isUnknown()) {
    isFeasible = true;
    return St;
  }

  if (isa<NonLVal>(Cond))
    return Assume(St, cast<NonLVal>(Cond), Assumption, isFeasible);
  else
    return Assume(St, cast<LVal>(Cond), Assumption, isFeasible);
}

const GRState* BasicConstraintManager::Assume(const GRState* St, LVal Cond,
                                            bool Assumption, bool& isFeasible) {
  St = AssumeAux(St, Cond, Assumption, isFeasible);
  // TF->EvalAssume(*this, St, Cond, Assumption, isFeasible)
  return St;
}

const GRState* BasicConstraintManager::AssumeAux(const GRState* St, LVal Cond,
                                            bool Assumption, bool& isFeasible) {
  BasicValueFactory& BasicVals = StateMgr.getBasicVals();

  switch (Cond.getSubKind()) {
  default:
    assert (false && "'Assume' not implemented for this LVal.");
    return St;

  case lval::SymbolValKind:
    if (Assumption)
      return AssumeSymNE(St, cast<lval::SymbolVal>(Cond).getSymbol(),
                         BasicVals.getZeroWithPtrWidth(), isFeasible);
    else
      return AssumeSymEQ(St, cast<lval::SymbolVal>(Cond).getSymbol(),
                         BasicVals.getZeroWithPtrWidth(), isFeasible);

  case lval::DeclValKind:
  case lval::FuncValKind:
  case lval::GotoLabelKind:
  case lval::StringLiteralValKind:
    isFeasible = Assumption;
    return St;

  case lval::FieldOffsetKind:
    return AssumeAux(St, cast<lval::FieldOffset>(Cond).getBase(),
                     Assumption, isFeasible);

  case lval::ArrayOffsetKind:
    return AssumeAux(St, cast<lval::ArrayOffset>(Cond).getBase(),
                     Assumption, isFeasible);

  case lval::ConcreteIntKind: {
    bool b = cast<lval::ConcreteInt>(Cond).getValue() != 0;
    isFeasible = b ? Assumption : !Assumption;
    return St;
  }
  } // end switch
}

const GRState*
BasicConstraintManager::Assume(const GRState* St, NonLVal Cond, bool Assumption,
                               bool& isFeasible) {
  St = AssumeAux(St, Cond, Assumption, isFeasible);
  // TF->EvalAssume() does nothing now.
  return St;
}

const GRState*
BasicConstraintManager::AssumeAux(const GRState* St,NonLVal Cond,
                                  bool Assumption, bool& isFeasible) {
  BasicValueFactory& BasicVals = StateMgr.getBasicVals();
  SymbolManager& SymMgr = StateMgr.getSymbolManager();

  switch (Cond.getSubKind()) {
  default:
    assert(false && "'Assume' not implemented for this NonLVal");

  case nonlval::SymbolValKind: {
    nonlval::SymbolVal& SV = cast<nonlval::SymbolVal>(Cond);
    SymbolID sym = SV.getSymbol();

    if (Assumption)
      return AssumeSymNE(St, sym, BasicVals.getValue(0, SymMgr.getType(sym)),
                         isFeasible);
    else
      return AssumeSymEQ(St, sym, BasicVals.getValue(0, SymMgr.getType(sym)),
                         isFeasible);
  }

  case nonlval::SymIntConstraintValKind:
    return
      AssumeSymInt(St, Assumption,
                   cast<nonlval::SymIntConstraintVal>(Cond).getConstraint(),
                   isFeasible);

  case nonlval::ConcreteIntKind: {
    bool b = cast<nonlval::ConcreteInt>(Cond).getValue() != 0;
    isFeasible = b ? Assumption : !Assumption;
    return St;
  }

  case nonlval::LValAsIntegerKind:
    return AssumeAux(St, cast<nonlval::LValAsInteger>(Cond).getLVal(),
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

  case BinaryOperator::GE:
    if (Assumption)
      return AssumeSymGE(St, C.getSymbol(), C.getInt(), isFeasible);
    else
      return AssumeSymLT(St, C.getSymbol(), C.getInt(), isFeasible);

  case BinaryOperator::LE:
    if (Assumption)
      return AssumeSymLE(St, C.getSymbol(), C.getInt(), isFeasible);
    else
      return AssumeSymGT(St, C.getSymbol(), C.getInt(), isFeasible);
  } // end switch
}



const GRState*
BasicConstraintManager::AssumeSymNE(const GRState* St, SymbolID sym,
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
BasicConstraintManager::AssumeSymEQ(const GRState* St, SymbolID sym,
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
BasicConstraintManager::AssumeSymLT(const GRState* St, SymbolID sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  // FIXME: For now have assuming x < y be the same as assuming sym != V;
  return AssumeSymNE(St, sym, V, isFeasible);
}

const GRState*
BasicConstraintManager::AssumeSymGT(const GRState* St, SymbolID sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  // FIXME: For now have assuming x > y be the same as assuming sym != V;
  return AssumeSymNE(St, sym, V, isFeasible);
}

const GRState*
BasicConstraintManager::AssumeSymGE(const GRState* St, SymbolID sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  // FIXME: Primitive logic for now.  Only reject a path if the value of
  //  sym is a constant X and !(X >= V).

  if (const llvm::APSInt* X = getSymVal(St, sym)) {
    isFeasible = *X >= V;
    return St;
  }

  isFeasible = true;
  return St;
}

const GRState*
BasicConstraintManager::AssumeSymLE(const GRState* St, SymbolID sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  // FIXME: Primitive logic for now.  Only reject a path if the value of
  //  sym is a constant X and !(X <= V).

  if (const llvm::APSInt* X = getSymVal(St, sym)) {
    isFeasible = *X <= V;
    return St;
  }

  isFeasible = true;
  return St;
}

static int ConstEqTyIndex = 0;
static int ConstNotEqTyIndex = 0;

namespace clang {
  template<>
  struct GRStateTrait<ConstNotEqTy> : public GRStatePartialTrait<ConstNotEqTy> {
    static inline void* GDMIndex() { return &ConstNotEqTyIndex; }  
  };
  
  template<>
  struct GRStateTrait<ConstEqTy> : public GRStatePartialTrait<ConstEqTy> {
    static inline void* GDMIndex() { return &ConstEqTyIndex; }  
  };
}

const GRState* BasicConstraintManager::AddEQ(const GRState* St, SymbolID sym,
                                             const llvm::APSInt& V) {
  // Create a new state with the old binding replaced.
  GRStateRef state(St, StateMgr);
  return state.set<ConstEqTy>(sym, &V);
}

const GRState* BasicConstraintManager::AddNE(const GRState* St, SymbolID sym,
                                             const llvm::APSInt& V) {
  GRState::IntSetTy::Factory ISetFactory(StateMgr.getAllocator());
  GRStateRef state(St, StateMgr);

  // First, retrieve the NE-set associated with the given symbol.
  ConstNotEqTy::data_type* T = state.get<ConstNotEqTy>(sym);
  GRState::IntSetTy S = T ? *T : ISetFactory.GetEmptySet();

  
  // Now add V to the NE set.
  S = ISetFactory.Add(S, &V);
  
  // Create a new state with the old binding replaced.
  return state.set<ConstNotEqTy>(sym, S);
}

const llvm::APSInt* BasicConstraintManager::getSymVal(const GRState* St,
                                                      SymbolID sym) {
  const ConstEqTy::data_type* T = St->get<ConstEqTy>(sym);
  return T ? *T : NULL;  
}

bool BasicConstraintManager::isNotEqual(const GRState* St, SymbolID sym, 
                                        const llvm::APSInt& V) const {

  // Retrieve the NE-set associated with the given symbol.
  const ConstNotEqTy::data_type* T = St->get<ConstNotEqTy>(sym);

  // See if V is present in the NE-set.
  return T ? T->contains(&V) : false;
}

bool BasicConstraintManager::isEqual(const GRState* St, SymbolID sym,
                                     const llvm::APSInt& V) const {
  // Retrieve the EQ-set associated with the given symbol.
  const ConstEqTy::data_type* T = St->get<ConstEqTy>(sym);
  // See if V is present in the EQ-set.
  return T ? **T == V : false;
}

const GRState* BasicConstraintManager::RemoveDeadBindings(const GRState* St,
                                        StoreManager::LiveSymbolsTy& LSymbols,
                                        StoreManager::DeadSymbolsTy& DSymbols) {
  GRStateRef state(St, StateMgr);
  ConstEqTy CE = state.get<ConstEqTy>();
  ConstEqTy::Factory& CEFactory = state.get_context<ConstEqTy>();

  for (ConstEqTy::iterator I = CE.begin(), E = CE.end(); I!=E; ++I) {
    SymbolID sym = I.getKey();        
    if (!LSymbols.count(sym)) {
      DSymbols.insert(sym);
      CE = CEFactory.Remove(CE, sym);
    }
  }
  state = state.set<ConstEqTy>(CE);

  ConstNotEqTy CNE = state.get<ConstNotEqTy>();
  ConstNotEqTy::Factory& CNEFactory = state.get_context<ConstNotEqTy>();

  for (ConstNotEqTy::iterator I = CNE.begin(), E = CNE.end(); I != E; ++I) {
    SymbolID sym = I.getKey();    
    if (!LSymbols.count(sym)) {
      DSymbols.insert(sym);
      CNE = CNEFactory.Remove(CNE, sym);
    }
  }
  
  return state.set<ConstNotEqTy>(CNE);
}

void BasicConstraintManager::print(const GRState* St, std::ostream& Out, 
                                   const char* nl, const char *sep) {
  // Print equality constraints.

  ConstEqTy CE = St->get<ConstEqTy>();

  if (!CE.isEmpty()) {
    Out << nl << sep << "'==' constraints:";

    for (ConstEqTy::iterator I = CE.begin(), E = CE.end(); I!=E; ++I) {
      Out << nl << " $" << I.getKey();
      llvm::raw_os_ostream OS(Out);
      OS << " : "   << *I.getData();
    }
  }

  // Print != constraints.
  
  ConstNotEqTy CNE = St->get<ConstNotEqTy>();
  
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
      
        Out << *J;
      }
    }
  }
}