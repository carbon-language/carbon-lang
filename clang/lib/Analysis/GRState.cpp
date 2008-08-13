//= GRState*cpp - Path-Sens. "State" for tracking valuues -----*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SymbolID, ExprBindKey, and GRState*
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/GRState.h"
#include "llvm/ADT/SmallSet.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"

using namespace clang;

bool GRState::isNotEqual(SymbolID sym, const llvm::APSInt& V) const {

  // Retrieve the NE-set associated with the given symbol.
  const ConstNotEqTy::data_type* T = ConstNotEq.lookup(sym);

  // See if V is present in the NE-set.
  return T ? T->contains(&V) : false;
}

bool GRState::isEqual(SymbolID sym, const llvm::APSInt& V) const {
  
  // Retrieve the EQ-set associated with the given symbol.
  const ConstEqTy::data_type* T = ConstEq.lookup(sym);
  
  // See if V is present in the EQ-set.
  return T ? **T == V : false;
}

const llvm::APSInt* GRState::getSymVal(SymbolID sym) const {
  ConstEqTy::data_type* T = ConstEq.lookup(sym);
  return T ? *T : NULL;  
}

const GRState*
GRStateManager::RemoveDeadBindings(const GRState* St, Stmt* Loc,
                                      const LiveVariables& Liveness,
                                      DeadSymbolsTy& DSymbols) {  
  
  // This code essentially performs a "mark-and-sweep" of the VariableBindings.
  // The roots are any Block-level exprs and Decls that our liveness algorithm
  // tells us are live.  We then see what Decls they may reference, and keep
  // those around.  This code more than likely can be made faster, and the
  // frequency of which this method is called should be experimented with
  // for optimum performance.  
  DRoots.clear();
  StoreManager::LiveSymbolsTy LSymbols;
  
  GRState NewSt = *St;

  // FIXME: Put this in environment.
  // Clean up the environment.
  
  // Drop bindings for subexpressions.
  NewSt.Env = EnvMgr.RemoveSubExprBindings(NewSt.Env);
  
  // Iterate over the block-expr bindings.

  for (GRState::beb_iterator I = St->beb_begin(), E = St->beb_end();
                                                    I!=E ; ++I) {    
    Expr* BlkExpr = I.getKey();
    
    if (Liveness.isLive(Loc, BlkExpr)) {
      RVal X = I.getData();
      
      if (isa<lval::DeclVal>(X)) {
        lval::DeclVal LV = cast<lval::DeclVal>(X);
        DRoots.push_back(LV.getDecl());
      }
      
      for (RVal::symbol_iterator SI = X.symbol_begin(), SE = X.symbol_end(); 
                                                        SI != SE; ++SI) {        
        LSymbols.insert(*SI);
      }
    }
    else {
      RVal X = I.getData();
      
      if (X.isUndef() && cast<UndefinedVal>(X).getData())
        continue;
      
      NewSt.Env = EnvMgr.RemoveBlkExpr(NewSt.Env, BlkExpr);
    }
  }

  // Clean up the store.
  DSymbols.clear();
  NewSt.St = StMgr->RemoveDeadBindings(St->getStore(), Loc, Liveness, DRoots,
                                       LSymbols, DSymbols);
  
  // Remove the dead symbols from the symbol tracker.
  for (GRState::ce_iterator I = St->ce_begin(), E=St->ce_end(); I!=E; ++I) {

    SymbolID sym = I.getKey();    
    
    if (!LSymbols.count(sym)) {
      DSymbols.insert(sym);
      NewSt.ConstEq = CEFactory.Remove(NewSt.ConstEq, sym);
    }
  }
  
  for (GRState::cne_iterator I = St->cne_begin(), E=St->cne_end(); I!=E;++I){
    
    SymbolID sym = I.getKey();
    
    if (!LSymbols.count(sym)) {
      DSymbols.insert(sym);
      NewSt.ConstNotEq = CNEFactory.Remove(NewSt.ConstNotEq, sym);
    }
  }
  
  return getPersistentState(NewSt);
}

const GRState* GRStateManager::SetRVal(const GRState* St, LVal LV,
                                             RVal V) {
  
  Store OldStore = St->getStore();
  Store NewStore = StMgr->SetRVal(OldStore, LV, V);
  
  if (NewStore == OldStore)
    return St;
  
  GRState NewSt = *St;
  NewSt.St = NewStore;
  return getPersistentState(NewSt);    
}

const GRState* GRStateManager::Unbind(const GRState* St, LVal LV) {
  Store OldStore = St->getStore();
  Store NewStore = StMgr->Remove(OldStore, LV);
  
  if (NewStore == OldStore)
    return St;
  
  GRState NewSt = *St;
  NewSt.St = NewStore;
  return getPersistentState(NewSt);    
}


const GRState* GRStateManager::AddNE(const GRState* St, SymbolID sym,
                                           const llvm::APSInt& V) {

  // First, retrieve the NE-set associated with the given symbol.
  GRState::ConstNotEqTy::data_type* T = St->ConstNotEq.lookup(sym);  
  GRState::IntSetTy S = T ? *T : ISetFactory.GetEmptySet();
  
  // Now add V to the NE set.
  S = ISetFactory.Add(S, &V);
  
  // Create a new state with the old binding replaced.
  GRState NewSt = *St;
  NewSt.ConstNotEq = CNEFactory.Add(NewSt.ConstNotEq, sym, S);
    
  // Get the persistent copy.
  return getPersistentState(NewSt);
}

const GRState* GRStateManager::AddEQ(const GRState* St, SymbolID sym,
                                           const llvm::APSInt& V) {

  // Create a new state with the old binding replaced.
  GRState NewSt = *St;
  NewSt.ConstEq = CEFactory.Add(NewSt.ConstEq, sym, &V);
  
  // Get the persistent copy.
  return getPersistentState(NewSt);
}

const GRState* GRStateManager::getInitialState() {

  GRState StateImpl(EnvMgr.getInitialEnvironment(),
                       StMgr->getInitialStore(),
                       GDMFactory.GetEmptyMap(),
                       CNEFactory.GetEmptyMap(),
                       CEFactory.GetEmptyMap());
  
  return getPersistentState(StateImpl);
}

const GRState* GRStateManager::getPersistentState(GRState& State) {
  
  llvm::FoldingSetNodeID ID;
  State.Profile(ID);  
  void* InsertPos;
  
  if (GRState* I = StateSet.FindNodeOrInsertPos(ID, InsertPos))
    return I;
  
  GRState* I = (GRState*) Alloc.Allocate<GRState>();
  new (I) GRState(State);  
  StateSet.InsertNode(I, InsertPos);
  return I;
}

void GRState::printDOT(std::ostream& Out,
                       Printer** Beg, Printer** End) const {
  print(Out, Beg, End, "\\l", "\\|");
}

void GRState::printStdErr(Printer** Beg, Printer** End) const {
  print(*llvm::cerr, Beg, End);
}  

void GRState::print(std::ostream& Out, Printer** Beg, Printer** End,
                    const char* nl, const char* sep) const {

  // Print Variable Bindings
  Out << "Variables:" << nl;
  
  bool isFirst = true;
  
  for (vb_iterator I = vb_begin(), E = vb_end(); I != E; ++I) {        
    
    if (isFirst) isFirst = false;
    else Out << nl;
    
    Out << ' ' << I.getKey()->getName() << " : ";
    I.getData().print(Out);
  }
  
  // Print Subexpression bindings.
  
  isFirst = true;
  
  for (seb_iterator I = seb_begin(), E = seb_end(); I != E; ++I) {        
    
    if (isFirst) {
      Out << nl << nl << "Sub-Expressions:" << nl;
      isFirst = false;
    }
    else { Out << nl; }
    
    Out << " (" << (void*) I.getKey() << ") ";
    I.getKey()->printPretty(Out);
    Out << " : ";
    I.getData().print(Out);
  }
  
  // Print block-expression bindings.
  
  isFirst = true;
  
  for (beb_iterator I = beb_begin(), E = beb_end(); I != E; ++I) {      

    if (isFirst) {
      Out << nl << nl << "Block-level Expressions:" << nl;
      isFirst = false;
    }
    else { Out << nl; }
    
    Out << " (" << (void*) I.getKey() << ") ";
    I.getKey()->printPretty(Out);
    Out << " : ";
    I.getData().print(Out);
  }
  
  // Print equality constraints.
  // FIXME: Make just another printer do this.
  
  if (!ConstEq.isEmpty()) {
  
    Out << nl << sep << "'==' constraints:";
  
    for (ConstEqTy::iterator I = ConstEq.begin(),
                             E = ConstEq.end();   I!=E; ++I) {
      
      Out << nl << " $" << I.getKey()
          << " : "   << I.getData()->toString();
    }
  }

  // Print != constraints.
  // FIXME: Make just another printer do this.
    
  if (!ConstNotEq.isEmpty()) {
  
    Out << nl << sep << "'!=' constraints:";
  
    for (ConstNotEqTy::iterator I  = ConstNotEq.begin(),
                                EI = ConstNotEq.end();   I != EI; ++I) {
    
      Out << nl << " $" << I.getKey() << " : ";
      isFirst = true;
    
      IntSetTy::iterator J = I.getData().begin(), EJ = I.getData().end();      
      
      for ( ; J != EJ; ++J) {        
        if (isFirst) isFirst = false;
        else Out << ", ";
      
        Out << (*J)->toString();
      }
    }
  }
  
  // Print checker-specific data. 
  for ( ; Beg != End ; ++Beg) (*Beg)->Print(Out, this, nl, sep);
}


//===----------------------------------------------------------------------===//
// Queries.
//===----------------------------------------------------------------------===//

bool GRStateManager::isEqual(const GRState* state, Expr* Ex,
                                const llvm::APSInt& Y) {
  RVal V = GetRVal(state, Ex);
  
  if (lval::ConcreteInt* X = dyn_cast<lval::ConcreteInt>(&V))
    return X->getValue() == Y;

  if (nonlval::ConcreteInt* X = dyn_cast<nonlval::ConcreteInt>(&V))
    return X->getValue() == Y;
    
  if (nonlval::SymbolVal* X = dyn_cast<nonlval::SymbolVal>(&V))
    return state->isEqual(X->getSymbol(), Y);
  
  if (lval::SymbolVal* X = dyn_cast<lval::SymbolVal>(&V))
    return state->isEqual(X->getSymbol(), Y);
  
  return false;
}
  
bool GRStateManager::isEqual(const GRState* state, Expr* Ex,
                                uint64_t x) {
  return isEqual(state, Ex, BasicVals.getValue(x, Ex->getType()));
}

//===----------------------------------------------------------------------===//
// "Assume" logic.
//===----------------------------------------------------------------------===//

const GRState* GRStateManager::Assume(const GRState* St, LVal Cond,
                                            bool Assumption, bool& isFeasible) {
  
  St = AssumeAux(St, Cond, Assumption, isFeasible);
  
  return isFeasible ? TF->EvalAssume(*this, St, Cond, Assumption, isFeasible)
                    : St;
}

const GRState* GRStateManager::AssumeAux(const GRState* St, LVal Cond,
                                          bool Assumption, bool& isFeasible) {
  
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
  }
}

const GRState* GRStateManager::Assume(const GRState* St, NonLVal Cond,
                                       bool Assumption, bool& isFeasible) {
  
  St = AssumeAux(St, Cond, Assumption, isFeasible);
  
  return isFeasible ? TF->EvalAssume(*this, St, Cond, Assumption, isFeasible)
  : St;
}

const GRState* GRStateManager::AssumeAux(const GRState* St, NonLVal Cond,
                                          bool Assumption, bool& isFeasible) {  
  switch (Cond.getSubKind()) {
    default:
      assert (false && "'Assume' not implemented for this NonLVal.");
      return St;
      
      
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
      
    case nonlval::LValAsIntegerKind: {
      return AssumeAux(St, cast<nonlval::LValAsInteger>(Cond).getLVal(),
                       Assumption, isFeasible);
    }
  }
}



const GRState* GRStateManager::AssumeSymInt(const GRState* St,
                                             bool Assumption,
                                             const SymIntConstraint& C,
                                             bool& isFeasible) {
  
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
  }
}

//===----------------------------------------------------------------------===//
// FIXME: This should go into a plug-in constraint engine.
//===----------------------------------------------------------------------===//

const GRState*
GRStateManager::AssumeSymNE(const GRState* St, SymbolID sym,
                               const llvm::APSInt& V, bool& isFeasible) {
  
  // First, determine if sym == X, where X != V.
  if (const llvm::APSInt* X = St->getSymVal(sym)) {
    isFeasible = *X != V;
    return St;
  }
  
  // Second, determine if sym != V.
  if (St->isNotEqual(sym, V)) {
    isFeasible = true;
    return St;
  }
  
  // If we reach here, sym is not a constant and we don't know if it is != V.
  // Make that assumption.
  
  isFeasible = true;
  return AddNE(St, sym, V);
}

const GRState*
GRStateManager::AssumeSymEQ(const GRState* St, SymbolID sym,
                               const llvm::APSInt& V, bool& isFeasible) {
  
  // First, determine if sym == X, where X != V.
  if (const llvm::APSInt* X = St->getSymVal(sym)) {
    isFeasible = *X == V;
    return St;
  }
  
  // Second, determine if sym != V.
  if (St->isNotEqual(sym, V)) {
    isFeasible = false;
    return St;
  }
  
  // If we reach here, sym is not a constant and we don't know if it is == V.
  // Make that assumption.
  
  isFeasible = true;
  return AddEQ(St, sym, V);
}

const GRState*
GRStateManager::AssumeSymLT(const GRState* St, SymbolID sym,
                               const llvm::APSInt& V, bool& isFeasible) {
  
  // FIXME: For now have assuming x < y be the same as assuming sym != V;
  return AssumeSymNE(St, sym, V, isFeasible);
}

const GRState*
GRStateManager::AssumeSymGT(const GRState* St, SymbolID sym,
                               const llvm::APSInt& V, bool& isFeasible) {
  
  // FIXME: For now have assuming x > y be the same as assuming sym != V;
  return AssumeSymNE(St, sym, V, isFeasible);
}

const GRState*
GRStateManager::AssumeSymGE(const GRState* St, SymbolID sym,
                               const llvm::APSInt& V, bool& isFeasible) {
  
  // FIXME: Primitive logic for now.  Only reject a path if the value of
  //  sym is a constant X and !(X >= V).
  
  if (const llvm::APSInt* X = St->getSymVal(sym)) {
    isFeasible = *X >= V;
    return St;
  }
  
  isFeasible = true;
  return St;
}

const GRState*
GRStateManager::AssumeSymLE(const GRState* St, SymbolID sym,
                               const llvm::APSInt& V, bool& isFeasible) {
  
  // FIXME: Primitive logic for now.  Only reject a path if the value of
  //  sym is a constant X and !(X <= V).
    
  if (const llvm::APSInt* X = St->getSymVal(sym)) {
    isFeasible = *X <= V;
    return St;
  }
  
  isFeasible = true;
  return St;
}

