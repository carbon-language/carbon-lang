//= ValueState.cpp - Path-Sens. "State" for tracking valuues -----*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines SymbolID, VarBindKey, and ValueState.
//
//===----------------------------------------------------------------------===//

#include "ValueState.h"

using namespace clang;

RValue ValueStateManager::GetValue(const StateTy& St, const LValue& LV) {
  switch (LV.getSubKind()) {
    case LValueDeclKind: {
      StateTy::VariableBindingsTy::TreeTy* T =
        St.getImpl()->VariableBindings.SlimFind(cast<LValueDecl>(LV).getDecl());
      
      return T ? T->getValue().second : InvalidValue();
    }
    default:
      assert (false && "Invalid LValue.");
      break;
  }
  
  return InvalidValue();
}

RValue ValueStateManager::GetValue(const StateTy& St, Stmt* S, bool* hasVal) {
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
        return GetValue(St, LValueDecl(cast<DeclRefExpr>(S)->getDecl()));
        
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
  
  StateTy::VariableBindingsTy::TreeTy* T =
    St.getImpl()->VariableBindings.SlimFind(S);
  
  if (T) {
    if (hasVal) *hasVal = true;
    return T->getValue().second;
  }
  else {
    if (hasVal) *hasVal = false;
    return InvalidValue();
  }
}

LValue ValueStateManager::GetLValue(const StateTy& St, Stmt* S) {
  
  while (ParenExpr* P = dyn_cast<ParenExpr>(S))
    S = P->getSubExpr();
  
  if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(S))
    return LValueDecl(DR->getDecl());
  
  return cast<LValue>(GetValue(St, S));
}


ValueStateManager::StateTy 
ValueStateManager::SetValue(StateTy St, Stmt* S, bool isBlkExpr,
                            const RValue& V) {
  
  assert (S);
  return V.isValid() ? Add(St, VarBindKey(S, isBlkExpr), V) : St;
}

ValueStateManager::StateTy
ValueStateManager::SetValue(StateTy St, const LValue& LV, const RValue& V) {
  
  switch (LV.getSubKind()) {
    case LValueDeclKind:        
      return V.isValid() ? Add(St, cast<LValueDecl>(LV).getDecl(), V)
                         : Remove(St, cast<LValueDecl>(LV).getDecl());
      
    default:
      assert ("SetValue for given LValue type not yet implemented.");
      return St;
  }
}

ValueStateManager::StateTy
ValueStateManager::Remove(StateTy St, VarBindKey K) {

  // Create a new state with the old binding removed.
  ValueStateImpl NewStateImpl = *St.getImpl();
  NewStateImpl.VariableBindings =
    VBFactory.Remove(NewStateImpl.VariableBindings, K);

  // Get the persistent copy.
  return getPersistentState(NewStateImpl);
}

ValueStateManager::StateTy
ValueStateManager::Add(StateTy St, VarBindKey K, const RValue& V) {
  
  // Create a new state with the old binding removed.
  ValueStateImpl NewStateImpl = *St.getImpl();
  NewStateImpl.VariableBindings =
    VBFactory.Add(NewStateImpl.VariableBindings, K, V);
  
  // Get the persistent copy.
  return getPersistentState(NewStateImpl);
}


ValueStateManager::StateTy
ValueStateManager::getInitialState() {

  // Create a state with empty variable bindings.
  ValueStateImpl StateImpl(VBFactory.GetEmptyMap(),
                           CNEFactory.GetEmptyMap());
  
  return getPersistentState(StateImpl);
}

ValueStateManager::StateTy
ValueStateManager::getPersistentState(const ValueStateImpl &State) {
  
  llvm::FoldingSetNodeID ID;
  State.Profile(ID);  
  void* InsertPos;  
  
  if (ValueStateImpl* I = StateSet.FindNodeOrInsertPos(ID, InsertPos))
    return I;
  
  ValueStateImpl* I = (ValueStateImpl*) Alloc.Allocate<ValueState>();
  new (I) ValueStateImpl(State);  
  StateSet.InsertNode(I, InsertPos);
  return I;
}
