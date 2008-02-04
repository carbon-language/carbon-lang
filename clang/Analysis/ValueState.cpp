#include "ValueState.h"

using namespace clang;

RValue ValueStateManager::GetValue(const StateTy& St, const LValue& LV) {
  switch (LV.getSubKind()) {
    case LValueDeclKind: {
      StateTy::TreeTy* T = St.SlimFind(cast<LValueDecl>(LV).getDecl()); 
      return T ? T->getValue().second : InvalidValue();
    }
    default:
      assert (false && "Invalid LValue.");
      break;
  }
  
  return InvalidValue();
}

RValue ValueStateManager::GetValue(const StateTy& St, Stmt* S) {
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
  
  StateTy::TreeTy* T = St.SlimFind(S);
  
  return T ? T->getValue().second : InvalidValue();
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
  return V.isValid() ? Factory.Add(St, ValueKey(S, isBlkExpr), V) : St;
}

ValueStateManager::StateTy
ValueStateManager::SetValue(StateTy St, const LValue& LV, const RValue& V) {
  
  switch (LV.getSubKind()) {
    case LValueDeclKind:        
      return V.isValid() ? Factory.Add(St, cast<LValueDecl>(LV).getDecl(), V)
      : Factory.Remove(St, cast<LValueDecl>(LV).getDecl());
      
    default:
      assert ("SetValue for given LValue type not yet implemented.");
      return St;
  }
}

ValueStateManager::StateTy ValueStateManager::Remove(StateTy St, ValueKey K) {
  return Factory.Remove(St, K);
}
  
