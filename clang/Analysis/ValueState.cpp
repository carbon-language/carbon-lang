//= ValueState.cpp - Path-Sens. "State" for tracking valuues -----*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines SymbolID, ExprBindKey, and ValueState.
//
//===----------------------------------------------------------------------===//

#include "ValueState.h"

using namespace clang;

bool ValueState::isNotEqual(SymbolID sym, const llvm::APSInt& V) const {
  // First, retrieve the NE-set associated with the given symbol.
  ConstantNotEqTy::TreeTy* T = Data->ConstantNotEq.SlimFind(sym);
  
  if (!T)
    return false;
  
  // Second, see if V is present in the NE-set.
  return T->getValue().second.contains(&V);
}

const llvm::APSInt* ValueState::getSymVal(SymbolID sym) const {
  ConstantEqTy::TreeTy* T = Data->ConstantEq.SlimFind(sym);
  return T ? T->getValue().second : NULL;  
}

ValueState
ValueStateManager::RemoveDeadBindings(ValueState St, Stmt* Loc,
                                      const LiveVariables& Liveness) {  
  
  // This code essentially performs a "mark-and-sweep" of the VariableBindings.
  // The roots are any Block-level exprs and Decls that our liveness algorithm
  // tells us are live.  We then see what Decls they may reference, and keep
  // those around.  This code more than likely can be made faster, and the
  // frequency of which this method is called should be experimented with
  // for optimum performance.
  
  llvm::SmallVector<ValueDecl*, 10> WList;

  for (StateTy::eb_iterator I = St.eb_begin(), E = St.eb_end(); I!=E ; ++I) {
    
    ExprBindKey K = I.getKey();
    
    // Remove old bindings for subexpressions.
    if (K.isSubExpr()) {
      St = Remove(St, K);
      continue;
    }
    
    assert (I.getKey().isBlkExpr());
    
    if (Liveness.isLive(Loc, K.getExpr())) {
      if (isa<lval::DeclVal>(I.getData())) {
        lval::DeclVal LV = cast<lval::DeclVal>(I.getData());
        WList.push_back(LV.getDecl());
      }
    }
    else
      St = Remove(St, K);
    
    continue;
  }

  for (StateTy::vb_iterator I = St.vb_begin(), E = St.vb_end(); I!=E ; ++I)
    if (Liveness.isLive(Loc, I.getKey()))
      WList.push_back(I.getKey());
  
  llvm::SmallPtrSet<ValueDecl*, 10> Marked;
  
  while (!WList.empty()) {
    ValueDecl* V = WList.back();
    WList.pop_back();
    
    if (Marked.count(V))
      continue;
    
    Marked.insert(V);
    
    if (V->getType()->isPointerType()) {
      const LValue& LV = cast<LValue>(GetValue(St, lval::DeclVal(V)));
      
      if (!isa<lval::DeclVal>(LV))
        continue;
      
      const lval::DeclVal& LVD = cast<lval::DeclVal>(LV);
      WList.push_back(LVD.getDecl());
    }    
  }
  
  for (StateTy::vb_iterator I = St.vb_begin(), E = St.vb_end(); I!=E ; ++I)
    if (!Marked.count(I.getKey()))
      St = Remove(St, I.getKey());
  
  return St;
}


RValue ValueStateManager::GetValue(const StateTy& St, const LValue& LV,
                                   QualType* T) {
  if (isa<UnknownVal>(LV))
    return UnknownVal();
  
  switch (LV.getSubKind()) {
    case lval::DeclValKind: {
      StateTy::VarBindingsTy::TreeTy* T =
      // FIXME: We should make lval::DeclVal only contain VarDecl
        St.getImpl()->VarBindings.SlimFind(
              cast<VarDecl>(cast<lval::DeclVal>(LV).getDecl()));
      
      return T ? T->getValue().second : UnknownVal();
    }
     
      // FIXME: We should bind how far a "ContentsOf" will go...
      
    case lval::SymbolValKind: {
      const lval::SymbolVal& SV = cast<lval::SymbolVal>(LV);
      assert (T);
      
      if (T->getTypePtr()->isPointerType())
        return lval::SymbolVal(SymMgr.getContentsOfSymbol(SV.getSymbol()));
      else
        return nonlval::SymbolVal(SymMgr.getContentsOfSymbol(SV.getSymbol()));
    }
      
    default:
      assert (false && "Invalid LValue.");
      break;
  }
  
  return UnknownVal();
}

ValueStateManager::StateTy
ValueStateManager::AddNE(StateTy St, SymbolID sym, const llvm::APSInt& V) {
  // First, retrieve the NE-set associated with the given symbol.
  ValueState::ConstantNotEqTy::TreeTy* T =
    St.getImpl()->ConstantNotEq.SlimFind(sym);    
  
  ValueState::IntSetTy S = T ? T->getValue().second : ISetFactory.GetEmptySet();
  
  // Now add V to the NE set.  
  S = ISetFactory.Add(S, &V);
  
  // Create a new state with the old binding replaced.
  ValueStateImpl NewStateImpl = *St.getImpl();
  NewStateImpl.ConstantNotEq = CNEFactory.Add(NewStateImpl.ConstantNotEq,
                                              sym, S);
    
  // Get the persistent copy.
  return getPersistentState(NewStateImpl);
}

ValueStateManager::StateTy
ValueStateManager::AddEQ(StateTy St, SymbolID sym, const llvm::APSInt& V) {
  // Create a new state with the old binding replaced.
  ValueStateImpl NewStateImpl = *St.getImpl();
  NewStateImpl.ConstantEq = CEFactory.Add(NewStateImpl.ConstantEq, sym, &V);
  
  // Get the persistent copy.
  return getPersistentState(NewStateImpl);
}

RValue ValueStateManager::GetValue(const StateTy& St, Expr* S, bool* hasVal) {
  for (;;) {
    switch (S->getStmtClass()) {
        
        // ParenExprs are no-ops.
        
      case Stmt::ParenExprClass:
        S = cast<ParenExpr>(S)->getSubExpr();
        continue;
        
        // DeclRefExprs can either evaluate to an LValue or a Non-LValue
        // (assuming an implicit "load") depending on the context.  In this
        // context we assume that we are retrieving the value contained
        // within the referenced variables.
        
      case Stmt::DeclRefExprClass:
        return GetValue(St, lval::DeclVal(cast<DeclRefExpr>(S)->getDecl()));
        
        // Integer literals evaluate to an RValue.  Simply retrieve the
        // RValue for the literal.
        
      case Stmt::IntegerLiteralClass:
        return NonLValue::GetValue(ValMgr, cast<IntegerLiteral>(S));
        
        // Casts where the source and target type are the same
        // are no-ops.  We blast through these to get the descendant
        // subexpression that has a value.
        
      case Stmt::ImplicitCastExprClass: {
        ImplicitCastExpr* C = cast<ImplicitCastExpr>(S);
        if (C->getType() == C->getSubExpr()->getType()) {
          S = C->getSubExpr();
          continue;
        }
        break;
      }
        
      case Stmt::CastExprClass: {
        CastExpr* C = cast<CastExpr>(S);
        if (C->getType() == C->getSubExpr()->getType()) {
          S = C->getSubExpr();
          continue;
        }
        break;
      }
        
        // Handle all other Stmt* using a lookup.
        
      default:
        break;
    };
    
    break;
  }
  
  StateTy::ExprBindingsTy::TreeTy* T =
    St.getImpl()->ExprBindings.SlimFind(S);
  
  if (T) {
    if (hasVal) *hasVal = true;
    return T->getValue().second;
  }
  else {
    if (hasVal) *hasVal = false;
    return UnknownVal();
  }
}

LValue ValueStateManager::GetLValue(const StateTy& St, Expr* E) {
  
  while (ParenExpr* P = dyn_cast<ParenExpr>(E))
    E = P->getSubExpr();
  
  if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(E))
    return lval::DeclVal(DR->getDecl());
  
  if (UnaryOperator* U = dyn_cast<UnaryOperator>(E))
    if (U->getOpcode() == UnaryOperator::Deref)
      return cast<LValue>(GetValue(St, U->getSubExpr()));
  
  return cast<LValue>(GetValue(St, E));
}


ValueStateManager::StateTy 
ValueStateManager::SetValue(StateTy St, Expr* E, bool isBlkExpr,
                            const RValue& V) {
  
  assert (E);
  return V.isKnown() ? Add(St, ExprBindKey(E, isBlkExpr), V) : St;
}

ValueStateManager::StateTy
ValueStateManager::SetValue(StateTy St, const LValue& LV, const RValue& V) {
  
  switch (LV.getSubKind()) {
    case lval::DeclValKind:        
      return V.isKnown()   // FIXME: Have DeclVal only contain VarDecl
        ? Add(St, cast<VarDecl>(cast<lval::DeclVal>(LV).getDecl()), V)
        : Remove(St, cast<VarDecl>(cast<lval::DeclVal>(LV).getDecl()));
      
    default:
      assert ("SetValue for given LValue type not yet implemented.");
      return St;
  }
}

ValueStateManager::StateTy
ValueStateManager::Add(StateTy St, ExprBindKey K, const RValue& V) {
  
  // Create a new state with the old binding removed.
  ValueStateImpl NewStateImpl = *St.getImpl();
  NewStateImpl.ExprBindings =
    EXFactory.Add(NewStateImpl.ExprBindings, K, V);
  
  // Get the persistent copy.
  return getPersistentState(NewStateImpl);
}

ValueStateManager::StateTy
ValueStateManager::Remove(StateTy St, ExprBindKey K) {
  
  // Create a new state with the old binding removed.
  ValueStateImpl NewStateImpl = *St.getImpl();
  NewStateImpl.ExprBindings =
    EXFactory.Remove(NewStateImpl.ExprBindings, K);
  
  // Get the persistent copy.
  return getPersistentState(NewStateImpl);
}

ValueStateManager::StateTy
ValueStateManager::Add(StateTy St, VarDecl* D, const RValue& V) {
  
  // Create a new state with the old binding removed.
  ValueStateImpl NewStateImpl = *St.getImpl();
  NewStateImpl.VarBindings =
    VBFactory.Add(NewStateImpl.VarBindings, D, V);
  
  // Get the persistent copy.
  return getPersistentState(NewStateImpl);
}

ValueStateManager::StateTy
ValueStateManager::Remove(StateTy St, VarDecl* D) {
  
  // Create a new state with the old binding removed.
  ValueStateImpl NewStateImpl = *St.getImpl();
  NewStateImpl.VarBindings =
    VBFactory.Remove(NewStateImpl.VarBindings, D);
  
  // Get the persistent copy.
  return getPersistentState(NewStateImpl);
}

ValueStateManager::StateTy
ValueStateManager::getInitialState() {

  // Create a state with empty variable bindings.
  ValueStateImpl StateImpl(EXFactory.GetEmptyMap(),
                           VBFactory.GetEmptyMap(),
                           CNEFactory.GetEmptyMap(),
                           CEFactory.GetEmptyMap());
  
  return getPersistentState(StateImpl);
}

ValueStateManager::StateTy
ValueStateManager::getPersistentState(const ValueStateImpl &State) {
  
  llvm::FoldingSetNodeID ID;
  State.Profile(ID);  
  void* InsertPos;  
  
  if (ValueStateImpl* I = StateSet.FindNodeOrInsertPos(ID, InsertPos))
    return I;
  
  ValueStateImpl* I = (ValueStateImpl*) Alloc.Allocate<ValueStateImpl>();
  new (I) ValueStateImpl(State);  
  StateSet.InsertNode(I, InsertPos);
  return I;
}
