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
#include "llvm/ADT/SmallSet.h"

using namespace clang;

bool ValueState::isNotEqual(SymbolID sym, const llvm::APSInt& V) const {

  // Retrieve the NE-set associated with the given symbol.
  ConstNotEqTy::TreeTy* T = Data->ConstNotEq.SlimFind(sym);

  // See if V is present in the NE-set.
  return T ? T->getValue().second.contains(&V) : false;
}

const llvm::APSInt* ValueState::getSymVal(SymbolID sym) const {
  ConstEqTy::TreeTy* T = Data->ConstEq.SlimFind(sym);
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
  llvm::SmallPtrSet<ValueDecl*, 10> Marked;  
  llvm::SmallSet<SymbolID, 20> MarkedSymbols;
  
  ValueStateImpl NewSt = *St;
  
  // Drop bindings for subexpressions.
  NewSt.SubExprBindings = EXFactory.GetEmptyMap();
  
  // Iterate over the block-expr bindings.

  for (ValueState::beb_iterator I = St.beb_begin(), E = St.beb_end();
                                                    I!=E ; ++I) {    
    Expr* BlkExpr = I.getKey();
    
    if (Liveness.isLive(Loc, BlkExpr)) {
      RVal X = I.getData();
      
      if (isa<lval::DeclVal>(X)) {
        lval::DeclVal LV = cast<lval::DeclVal>(X);
        WList.push_back(LV.getDecl());
      }
      
      for (RVal::symbol_iterator SI = X.symbol_begin(), SE = X.symbol_end(); 
                                                        SI != SE; ++SI) {        
        MarkedSymbols.insert(*SI);
      }
    }
    else {
      RVal X = I.getData();
      
      if (X.isUninit() && cast<UninitializedVal>(X).getData())
        continue;
      
      NewSt.BlockExprBindings = Remove(NewSt, BlkExpr);
    }
  }

  // Iterate over the variable bindings.

  for (ValueState::vb_iterator I = St.vb_begin(), E = St.vb_end(); I!=E ; ++I)
    if (Liveness.isLive(Loc, I.getKey())) {
      WList.push_back(I.getKey());
      
      RVal X = I.getData();
      
      for (RVal::symbol_iterator SI = X.symbol_begin(), SE = X.symbol_end(); 
           SI != SE; ++SI) {        
        MarkedSymbols.insert(*SI);
      }
    }

  // Perform the mark-and-sweep.

  while (!WList.empty()) {
    
    ValueDecl* V = WList.back();
    WList.pop_back();
    
    if (Marked.count(V))
      continue;
    
    Marked.insert(V);
    
    if (V->getType()->isPointerType()) {
      
      RVal X = GetRVal(St, lval::DeclVal(cast<VarDecl>(V)));      
      
      if (X.isUnknownOrUninit())
        continue;
      
      LVal LV = cast<LVal>(X);
      
      for (RVal::symbol_iterator SI = LV.symbol_begin(), SE = LV.symbol_end();
                                                         SI != SE; ++SI) {
        MarkedSymbols.insert(*SI);
      }
      
      if (!isa<lval::DeclVal>(LV))
        continue;
      
      const lval::DeclVal& LVD = cast<lval::DeclVal>(LV);
      WList.push_back(LVD.getDecl());
    }    
  }
  
  // Remove dead variable bindings.
  for (ValueState::vb_iterator I = St.vb_begin(), E = St.vb_end(); I!=E ; ++I)
    if (!Marked.count(I.getKey()))
      NewSt.VarBindings = Remove(NewSt, I.getKey());
  
  // Remove dead symbols.
  for (ValueState::ce_iterator I = St.ce_begin(), E=St.ce_end(); I!=E; ++I)
    if (!MarkedSymbols.count(I.getKey()))
      NewSt.ConstEq = CEFactory.Remove(NewSt.ConstEq, I.getKey());
  
  for (ValueState::cne_iterator I = St.cne_begin(), E=St.cne_end(); I!=E; ++I)
    if (!MarkedSymbols.count(I.getKey()))
      NewSt.ConstNotEq = CNEFactory.Remove(NewSt.ConstNotEq, I.getKey());
  
  return getPersistentState(NewSt);
}


RVal ValueStateManager::GetRVal(ValueState St, const LVal& LV, QualType T) {
  
  if (isa<UnknownVal>(LV))
    return UnknownVal();
  
  assert (!isa<UninitializedVal>(LV));
  
  switch (LV.getSubKind()) {
    case lval::DeclValKind: {
      ValueState::VarBindingsTy::TreeTy* T =
        St->VarBindings.SlimFind(cast<lval::DeclVal>(LV).getDecl());
      
      return T ? T->getValue().second : UnknownVal();
    }
     
      // FIXME: We should limit how far a "ContentsOf" will go...
      
    case lval::SymbolValKind: {
      const lval::SymbolVal& SV = cast<lval::SymbolVal>(LV);
      assert (T.getTypePtr());
      
      // Punt on "symbolic" function pointers.
      if (T->isFunctionType())
        return UnknownVal();
      
      if (T->isPointerType())
        return lval::SymbolVal(SymMgr.getContentsOfSymbol(SV.getSymbol()));
      else
        return nonlval::SymbolVal(SymMgr.getContentsOfSymbol(SV.getSymbol()));
    }
      
    default:
      assert (false && "Invalid LVal.");
      break;
  }
  
  return UnknownVal();
}

ValueState
ValueStateManager::AddNE(ValueState St, SymbolID sym, const llvm::APSInt& V) {

  // First, retrieve the NE-set associated with the given symbol.
  ValueState::ConstNotEqTy::TreeTy* T = St->ConstNotEq.SlimFind(sym);  
  ValueState::IntSetTy S = T ? T->getValue().second : ISetFactory.GetEmptySet();
  
  // Now add V to the NE set.
  S = ISetFactory.Add(S, &V);
  
  // Create a new state with the old binding replaced.
  ValueStateImpl NewSt = *St;
  NewSt.ConstNotEq = CNEFactory.Add(NewSt.ConstNotEq, sym, S);
    
  // Get the persistent copy.
  return getPersistentState(NewSt);
}

ValueState
ValueStateManager::AddEQ(ValueState St, SymbolID sym, const llvm::APSInt& V) {

  // Create a new state with the old binding replaced.
  ValueStateImpl NewSt = *St;
  NewSt.ConstEq = CEFactory.Add(NewSt.ConstEq, sym, &V);
  
  // Get the persistent copy.
  return getPersistentState(NewSt);
}

RVal ValueStateManager::GetRVal(ValueState St, Expr* E) {

  for (;;) {
    
    switch (E->getStmtClass()) {

      case Stmt::AddrLabelExprClass:        
        return LVal::MakeVal(cast<AddrLabelExpr>(E));
        
        // ParenExprs are no-ops.
        
      case Stmt::ParenExprClass:        
        E = cast<ParenExpr>(E)->getSubExpr();
        continue;
        
        // DeclRefExprs can either evaluate to an LVal or a Non-LVal
        // (assuming an implicit "load") depending on the context.  In this
        // context we assume that we are retrieving the value contained
        // within the referenced variables.
        
      case Stmt::DeclRefExprClass: {
        
        ValueDecl* D = cast<DeclRefExpr>(E)->getDecl();
        
        if (VarDecl* VD = dyn_cast<VarDecl>(D)) {          
          return GetRVal(St, lval::DeclVal(VD));          
        }
        else if (EnumConstantDecl* ED = dyn_cast<EnumConstantDecl>(D)) {
          
          // FIXME: Do we need to cache a copy of this enum, since it
          // already has persistent storage?  We do this because we
          // are comparing states using pointer equality.  Perhaps there is
          // a better way, since APInts are fairly lightweight.
          llvm::APSInt X = ED->getInitVal();
          
          // FIXME: This is a hack.  The APSInt inside the EnumConstantDecl
          //  might not match the signedness of the DeclRefExpr.  We hack
          //  a workaround here.  Should be fixed elsewhere.
          if (E->getType()->isUnsignedIntegerType() != X.isUnsigned())
            X.setIsUnsigned(!X.isUnsigned());
          
          return nonlval::ConcreteInt(ValMgr.getValue(X));          
        }
        else if (FunctionDecl* FD = dyn_cast<FunctionDecl>(D))
          return lval::FuncVal(FD);
        
        assert (false &&
                "ValueDecl support for this ValueDecl not implemented.");
        
        return UnknownVal();
      }

        // Casts where the source and target type are the same
        // are no-ops.  We blast through these to get the descendant
        // subexpression that has a value.
        
      case Stmt::ImplicitCastExprClass: {

        ImplicitCastExpr* C = cast<ImplicitCastExpr>(E);
        QualType CT = C->getType();
        
        if (CT->isVoidType())
          return UnknownVal();
          
        QualType ST = C->getSubExpr()->getType();
        
        if (CT == ST || (CT->isPointerType() && ST->isFunctionType())) {
          E = C->getSubExpr();
          continue;
        }
        
        break;
      }
        
      case Stmt::CastExprClass: {
        CastExpr* C = cast<CastExpr>(E);
        QualType CT = C->getType();
        QualType ST = C->getSubExpr()->getType();
        
        if (CT->isVoidType())
          return UnknownVal();
        
        if (CT == ST || (CT->isPointerType() && ST->isFunctionType())) {
          E = C->getSubExpr();
          continue;
        }
        
        break;
      }
        
      case Stmt::UnaryOperatorClass: {
        
        UnaryOperator* U = cast<UnaryOperator>(E);
        
        if (U->getOpcode() == UnaryOperator::Plus) {
          E = U->getSubExpr();
          continue;
        }
        
        break;
      }
          
        // Handle all other Expr* using a lookup.
        
      default:
        break;
    };
    
    break;
  }
  
  ValueState::ExprBindingsTy::TreeTy* T = St->SubExprBindings.SlimFind(E);
  
  return T ? T->getValue().second : GetBlkExprRVal(St, E);
}

RVal ValueStateManager::GetBlkExprRVal(ValueState St, Expr* E) {
  
  assert (!isa<ParenExpr>(E));

  ValueState::ExprBindingsTy::TreeTy* T = St->BlockExprBindings.SlimFind(E);    
  return T ? T->getValue().second : UnknownVal();
}

RVal ValueStateManager::GetLVal(ValueState St, Expr* E) {
  
  E = E->IgnoreParens();

  if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(E)) {
    ValueDecl* VD = DR->getDecl();
    
    if (FunctionDecl* FD = dyn_cast<FunctionDecl>(VD))
      return lval::FuncVal(FD);
    else
      return lval::DeclVal(cast<VarDecl>(DR->getDecl()));
  }
  
  if (UnaryOperator* U = dyn_cast<UnaryOperator>(E))
    if (U->getOpcode() == UnaryOperator::Deref) {
      E = U->getSubExpr()->IgnoreParens();
        
      if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(E)) {
        lval::DeclVal X(cast<VarDecl>(DR->getDecl()));
        return GetRVal(St, X);
      }
      else
        return GetRVal(St, E);
    }
        
  return GetRVal(St, E);
}

ValueState 
ValueStateManager::SetRVal(ValueState St, Expr* E, bool isBlkExpr, RVal V) {
  
  assert (E);

  if (V.isUnknown())
    return St;
  
  ValueStateImpl NewSt = *St;
  
  if (isBlkExpr) {
    NewSt.BlockExprBindings = EXFactory.Add(NewSt.BlockExprBindings, E, V);
  }
  else {
    NewSt.SubExprBindings = EXFactory.Add(NewSt.SubExprBindings, E, V);
  }

  return getPersistentState(NewSt);
}

ValueState
ValueStateManager::SetRVal(ValueState St, LVal LV, RVal V) {
  
  switch (LV.getSubKind()) {
      
    case lval::DeclValKind:        
      return V.isUnknown()
             ? UnbindVar(St, cast<lval::DeclVal>(LV).getDecl())
             : BindVar(St, cast<lval::DeclVal>(LV).getDecl(), V);
      
    default:
      assert ("SetRVal for given LVal type not yet implemented.");
      return St;
  }
}

ValueState ValueStateManager::BindVar(ValueState St, VarDecl* D, RVal V) {
  
  // Create a new state with the old binding removed.
  ValueStateImpl NewSt = *St;
  NewSt.VarBindings = VBFactory.Add(NewSt.VarBindings, D, V);
  
  // Get the persistent copy.
  return getPersistentState(NewSt);
}

ValueState ValueStateManager::UnbindVar(ValueState St, VarDecl* D) {
  
  // Create a new state with the old binding removed.
  ValueStateImpl NewSt = *St;
  NewSt.VarBindings = VBFactory.Remove(NewSt.VarBindings, D);
  
  // Get the persistent copy.
  return getPersistentState(NewSt);
}

ValueState ValueStateManager::getInitialState() {

  // Create a state with empty variable bindings.
  ValueStateImpl StateImpl(EXFactory.GetEmptyMap(),
                           VBFactory.GetEmptyMap(),
                           CNEFactory.GetEmptyMap(),
                           CEFactory.GetEmptyMap());
  
  return getPersistentState(StateImpl);
}

ValueState ValueStateManager::getPersistentState(const ValueStateImpl &State) {
  
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

void ValueState::printDOT(std::ostream& Out) const {

  // Print Variable Bindings
  Out << "Variables:\\l";
  
  bool isFirst = true;
  
  for (vb_iterator I = vb_begin(), E = vb_end(); I != E; ++I) {        
    
    if (isFirst) isFirst = false;
    else Out << "\\l";
    
    Out << ' ' << I.getKey()->getName() << " : ";
    I.getData().print(Out);
  }
  
  // Print Subexpression bindings.
  
  isFirst = true;
  
  for (seb_iterator I = seb_begin(), E = seb_end(); I != E; ++I) {        
    
    if (isFirst) {
      Out << "\\l\\lSub-Expressions:\\l";
      isFirst = false;
    }
    else { Out << "\\l"; }
    
    Out << " (" << (void*) I.getKey() << ") ";
    I.getKey()->printPretty(Out);
    Out << " : ";
    I.getData().print(Out);
  }
  
  // Print block-expression bindings.
  
  isFirst = true;
  
  for (beb_iterator I = beb_begin(), E = beb_end(); I != E; ++I) {      

    if (isFirst) {
      Out << "\\l\\lBlock-level Expressions:\\l";
      isFirst = false;
    }
    else { Out << "\\l"; }
    
    Out << " (" << (void*) I.getKey() << ") ";
    I.getKey()->printPretty(Out);
    Out << " : ";
    I.getData().print(Out);
  }
  
  // Print equality constraints.
  
  if (!Data->ConstEq.isEmpty()) {
  
    Out << "\\l\\|'==' constraints:";
  
    for (ConstEqTy::iterator I = Data->ConstEq.begin(),
                             E = Data->ConstEq.end();   I!=E; ++I) {
      
      Out << "\\l $" << I.getKey()
          << " : "   << I.getData()->toString();
    }
  }

  // Print != constraints.
    
  if (!Data->ConstNotEq.isEmpty()) {
  
    Out << "\\l\\|'!=' constraints:";
  
    for (ConstNotEqTy::iterator I  = Data->ConstNotEq.begin(),
                                EI = Data->ConstNotEq.end();   I != EI; ++I) {
    
      Out << "\\l $" << I.getKey() << " : ";
      isFirst = true;
    
      IntSetTy::iterator J = I.getData().begin(), EJ = I.getData().end();      
      
      for ( ; J != EJ; ++J) {        
        if (isFirst) isFirst = false;
        else Out << ", ";
      
        Out << (*J)->toString();
      }
    }
  }
}
