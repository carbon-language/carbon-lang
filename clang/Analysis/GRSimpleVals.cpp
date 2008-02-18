// GRSimpleVals.cpp - Transfer functions for tracking simple values -*- C++ -*--
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines GRSimpleVals, a sub-class of GRTransferFuncs that
//  provides transfer functions for performing simple value tracking with
//  limited support for symbolics.
//
//===----------------------------------------------------------------------===//

#include "GRSimpleVals.h"
#include "clang/Basic/Diagnostic.h"

using namespace clang;

namespace clang {
  void RunGRSimpleVals(CFG& cfg, FunctionDecl& FD, ASTContext& Ctx,
                      Diagnostic& Diag, bool Visualize) {
    
    if (Diag.hasErrorOccurred())
      return;
    
    GRCoreEngine<GRExprEngine> Engine(cfg, FD, Ctx);
    GRExprEngine* CheckerState = &Engine.getCheckerState();
    GRSimpleVals GRSV;
    CheckerState->setTransferFunctions(GRSV);
    
    // Execute the worklist algorithm.
    Engine.ExecuteWorkList(10000);
    
    // Look for explicit-Null dereferences and warn about them.
    for (GRExprEngine::null_iterator I=CheckerState->null_begin(),
         E=CheckerState->null_end(); I!=E; ++I) {
      
      const PostStmt& L = cast<PostStmt>((*I)->getLocation());
      Expr* Exp = cast<Expr>(L.getStmt());
      
      Diag.Report(FullSourceLoc(Exp->getExprLoc(), Ctx.getSourceManager()),
                  diag::chkr_null_deref_after_check);
    }
        
#ifndef NDEBUG
    if (Visualize) CheckerState->ViewGraph();
#endif  
  }
} // end clang namespace

//===----------------------------------------------------------------------===//
// Transfer function for Casts.
//===----------------------------------------------------------------------===//

RValue GRSimpleVals::EvalCast(ValueManager& ValMgr, NonLValue X,
                              Expr* CastExpr) {
  
  assert (!isa<UnknownVal>(X) && !isa<UninitializedVal>(X));
  
  if (!isa<nonlval::ConcreteInt>(X))
    return UnknownVal();
  
  llvm::APSInt V = cast<nonlval::ConcreteInt>(X).getValue();
  QualType T = CastExpr->getType();
  V.setIsUnsigned(T->isUnsignedIntegerType() || T->isPointerType());
  V.extOrTrunc(ValMgr.getContext().getTypeSize(T, CastExpr->getLocStart()));
  
  if (CastExpr->getType()->isPointerType())
    return lval::ConcreteInt(ValMgr.getValue(V));
  else
    return nonlval::ConcreteInt(ValMgr.getValue(V));
}

// Casts.

RValue GRSimpleVals::EvalCast(ValueManager& ValMgr, LValue X, Expr* CastExpr) {
  
  assert (!isa<UnknownVal>(X) && !isa<UninitializedVal>(X));

  if (CastExpr->getType()->isPointerType())
    return X;
  
  assert (CastExpr->getType()->isIntegerType());
  
  if (!isa<lval::ConcreteInt>(X))
    return UnknownVal();
  
  llvm::APSInt V = cast<lval::ConcreteInt>(X).getValue();
  QualType T = CastExpr->getType();
  V.setIsUnsigned(T->isUnsignedIntegerType() || T->isPointerType());
  V.extOrTrunc(ValMgr.getContext().getTypeSize(T, CastExpr->getLocStart()));

  return nonlval::ConcreteInt(ValMgr.getValue(V));
}

// Unary operators.

NonLValue GRSimpleVals::EvalMinus(ValueManager& ValMgr, UnaryOperator* U,
                                  NonLValue X) {
  
  assert (!isa<UnknownVal>(X) && !isa<UninitializedVal>(X));
  
  switch (X.getSubKind()) {
    case nonlval::ConcreteIntKind:
      return cast<nonlval::ConcreteInt>(X).EvalMinus(ValMgr, U);
    default:
      return cast<NonLValue>(UnknownVal());
  }
}

NonLValue GRSimpleVals::EvalComplement(ValueManager& ValMgr, NonLValue X) {

  assert (!isa<UnknownVal>(X) && !isa<UninitializedVal>(X));
  
  switch (X.getSubKind()) {
    case nonlval::ConcreteIntKind:
      return cast<nonlval::ConcreteInt>(X).EvalComplement(ValMgr);
    default:
      return cast<NonLValue>(UnknownVal());
  }
}

// Binary operators.

NonLValue GRSimpleVals::EvalBinaryOp(ValueManager& ValMgr,
                                     BinaryOperator::Opcode Op,
                                     NonLValue LHS, NonLValue RHS)  {
  
  assert (!isa<UnknownVal>(LHS) && !isa<UninitializedVal>(LHS));
  assert (!isa<UnknownVal>(RHS) && !isa<UninitializedVal>(RHS));
  
  while(1) {
    
    switch (LHS.getSubKind()) {
      default:
        return cast<NonLValue>(UnknownVal());
        
      case nonlval::ConcreteIntKind:
        
        if (isa<nonlval::ConcreteInt>(RHS)) {        
          const nonlval::ConcreteInt& LHS_CI = cast<nonlval::ConcreteInt>(LHS);
          const nonlval::ConcreteInt& RHS_CI = cast<nonlval::ConcreteInt>(RHS);
          return LHS_CI.EvalBinaryOp(ValMgr, Op, RHS_CI);
        }
        else if(isa<UnknownVal>(RHS))
          return cast<NonLValue>(UnknownVal());
        else {
          NonLValue tmp = RHS;
          RHS = LHS;
          LHS = tmp;
          continue;
        }
        
      case nonlval::SymbolValKind: {
        if (isa<nonlval::ConcreteInt>(RHS)) {
          const SymIntConstraint& C =
            ValMgr.getConstraint(cast<nonlval::SymbolVal>(LHS).getSymbol(), Op,
                                 cast<nonlval::ConcreteInt>(RHS).getValue());
          
          return nonlval::SymIntConstraintVal(C);
        }
        else
          return cast<NonLValue>(UnknownVal());
      }
    }
  }
}


// Binary Operators (except assignments and comma).

RValue GRSimpleVals::EvalBinaryOp(ValueManager& ValMgr,
                                  BinaryOperator::Opcode Op,
                                  LValue LHS, LValue RHS) {
  
  assert (!isa<UnknownVal>(LHS) && !isa<UninitializedVal>(LHS));
  assert (!isa<UnknownVal>(RHS) && !isa<UninitializedVal>(RHS));
  
  switch (Op) {
    default:
      return UnknownVal();
      
    case BinaryOperator::EQ:
      return EvalEQ(ValMgr, LHS, RHS);
      
    case BinaryOperator::NE:
      return EvalNE(ValMgr, LHS, RHS);      
  }
}

// Pointer arithmetic.

LValue GRSimpleVals::EvalBinaryOp(ValueManager& ValMgr,
                                  BinaryOperator::Opcode Op,
                                  LValue LHS, NonLValue RHS) {
  
  assert (!isa<UnknownVal>(LHS) && !isa<UninitializedVal>(LHS));
  assert (!isa<UnknownVal>(RHS) && !isa<UninitializedVal>(RHS));
  
  return cast<LValue>(UnknownVal());
}

// Equality operators for LValues.


NonLValue GRSimpleVals::EvalEQ(ValueManager& ValMgr, LValue LHS, LValue RHS) {
  
  assert (!isa<UnknownVal>(LHS) && !isa<UninitializedVal>(LHS));
  assert (!isa<UnknownVal>(RHS) && !isa<UninitializedVal>(RHS));
  
  switch (LHS.getSubKind()) {
    default:
      assert(false && "EQ not implemented for this LValue.");
      return cast<NonLValue>(UnknownVal());
      
    case lval::ConcreteIntKind:
      if (isa<lval::ConcreteInt>(RHS)) {
        bool b = cast<lval::ConcreteInt>(LHS).getValue() ==
                 cast<lval::ConcreteInt>(RHS).getValue();
        
        return NonLValue::GetIntTruthValue(ValMgr, b);
      }
      else if (isa<lval::SymbolVal>(RHS)) {
        
        const SymIntConstraint& C =
          ValMgr.getConstraint(cast<lval::SymbolVal>(RHS).getSymbol(),
                               BinaryOperator::EQ,
                               cast<lval::ConcreteInt>(LHS).getValue());
        
        return nonlval::SymIntConstraintVal(C);
      }
      
      break;
      
      case lval::SymbolValKind: {
        if (isa<lval::ConcreteInt>(RHS)) {          
          const SymIntConstraint& C =
            ValMgr.getConstraint(cast<lval::SymbolVal>(LHS).getSymbol(),
                                 BinaryOperator::EQ,
                                 cast<lval::ConcreteInt>(RHS).getValue());
          
          return nonlval::SymIntConstraintVal(C);
        }
        
        // FIXME: Implement unification
        return cast<NonLValue>(UnknownVal());
          //assert (!isa<lval::SymbolVal>(RHS) && "FIXME: Implement unification.");
        
        break;
      }
      
      case lval::DeclValKind:
      
        if (isa<lval::DeclVal>(RHS)) {        
          bool b = cast<lval::DeclVal>(LHS) == cast<lval::DeclVal>(RHS);
          return NonLValue::GetIntTruthValue(ValMgr, b);
        }
      
        break;
  }
  
  return NonLValue::GetIntTruthValue(ValMgr, false);
}

NonLValue GRSimpleVals::EvalNE(ValueManager& ValMgr, LValue LHS, LValue RHS) {
  
  assert (!isa<UnknownVal>(LHS) && !isa<UninitializedVal>(LHS));
  assert (!isa<UnknownVal>(RHS) && !isa<UninitializedVal>(RHS));

  switch (LHS.getSubKind()) {
    default:
      assert(false && "NE not implemented for this LValue.");
      return cast<NonLValue>(UnknownVal());
      
    case lval::ConcreteIntKind:
      if (isa<lval::ConcreteInt>(RHS)) {
        bool b = cast<lval::ConcreteInt>(LHS).getValue() !=
                 cast<lval::ConcreteInt>(RHS).getValue();
        
        return NonLValue::GetIntTruthValue(ValMgr, b);
      }
      else if (isa<lval::SymbolVal>(RHS)) {        
        const SymIntConstraint& C =
          ValMgr.getConstraint(cast<lval::SymbolVal>(RHS).getSymbol(),
                               BinaryOperator::NE,
                               cast<lval::ConcreteInt>(LHS).getValue());
        
        return nonlval::SymIntConstraintVal(C);
      }
      
      break;
      
      case lval::SymbolValKind: {
        if (isa<lval::ConcreteInt>(RHS)) {          
          const SymIntConstraint& C =
            ValMgr.getConstraint(cast<lval::SymbolVal>(LHS).getSymbol(),
                                 BinaryOperator::NE,
                                 cast<lval::ConcreteInt>(RHS).getValue());
          
          return nonlval::SymIntConstraintVal(C);
        }
        
        assert (!isa<lval::SymbolVal>(RHS) && "FIXME: Implement sym !=.");
        
        break;
      }
      
      case lval::DeclValKind:
        if (isa<lval::DeclVal>(RHS)) {        
          bool b = cast<lval::DeclVal>(LHS) == cast<lval::DeclVal>(RHS);
          return NonLValue::GetIntTruthValue(ValMgr, b);
        }
      
      break;
  }
  
  return NonLValue::GetIntTruthValue(ValMgr, true);
}
