// GRSimpleVals.cpp - Transfer functions for tracking simple values -*- C++ -*--
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines GRSimpleVals, a sub-class of GRTransferFuncs that
//  provides transfer functions for performing simple value tracking with
//  limited support for symbolics.
//
//===----------------------------------------------------------------------===//

#include "GRSimpleVals.h"
#include "BasicObjCFoundationChecks.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/PathSensitive/ValueState.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "llvm/Support/Compiler.h"
#include <sstream>

using namespace clang;

//===----------------------------------------------------------------------===//
// Bug Descriptions.
//===----------------------------------------------------------------------===//

namespace {
  
class VISIBILITY_HIDDEN NullDeref : public BugDescription {
public:
  virtual const char* getName() const {
    return "null dereference";
  }
  virtual const char* getDescription() const {
    return "Dereference of null pointer.";
  }
};
  
class VISIBILITY_HIDDEN UndefDeref : public BugDescription {
public:
  virtual const char* getName() const {
    return "bad dereference";
  }
  virtual const char* getDescription() const {
    return "Dereference of undefined value.";
  }
};
  
class VISIBILITY_HIDDEN UndefBranch : public BugDescription {
public:
  virtual const char* getName() const {
    return "uninitialized value";
  }
  virtual const char* getDescription() const {
    return "Branch condition evaluates to an uninitialized value.";
  }
};

class VISIBILITY_HIDDEN DivZero : public BugDescription {
public:
  virtual const char* getName() const {
    return "divide-by-zero";
  }
  virtual const char* getDescription() const {
    return "Division by zero/undefined value.";
  }
};
  
class VISIBILITY_HIDDEN UndefResult : public BugDescription {
public:
  virtual const char* getName() const {
    return "undefined result";
  }
  virtual const char* getDescription() const {
    return "Result of operation is undefined.";
  }
};
  
class VISIBILITY_HIDDEN BadCall : public BugDescription {
public:
  virtual const char* getName() const {
    return "invalid function call";
  }
  virtual const char* getDescription() const {
    return "Called function is a NULL or undefined function pointer value.";
  }
};
  
class VISIBILITY_HIDDEN BadArg : public BugDescription {
public:
  virtual const char* getName() const {
    return "bad argument";
  }
  virtual const char* getDescription() const {
    return "Pass-by-value argument in function is undefined.";
  }
};

class VISIBILITY_HIDDEN BadMsgExprArg : public BugDescription {
public:
  virtual const char* getName() const {
    return "bad receiver";
  }
  virtual const char* getDescription() const {
    return "Pass-by-value argument in message expression is undefined.";
  }
};

class VISIBILITY_HIDDEN BadReceiver : public BugDescription {
  SourceRange R;
public:
  BadReceiver(ExplodedNode<ValueState>* N) {
    Stmt *S = cast<PostStmt>(N->getLocation()).getStmt();
    Expr* E = cast<ObjCMessageExpr>(S)->getReceiver();
    assert (E && "Receiver cannot be NULL");
    R = E->getSourceRange();
  }
  
  virtual const char* getName() const {
    return "invalid message expression";
  }
  virtual const char* getDescription() const {
    return "Receiver in message expression is an uninitialized value.";
  }
  
  virtual void getRanges(const SourceRange*& B, const SourceRange*& E) const {
    B = &R;
    E = B+1;
  }
};
  
class VISIBILITY_HIDDEN RetStack : public BugDescription {
public:
  virtual const char* getName() const {
    return "return of stack address";
  }
  virtual const char* getDescription() const {
    return "Address of stack-allocated variable returned.";
  }
};
  
} // end anonymous namespace
  
//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//
  
template <typename ITERATOR> static inline
ExplodedNode<ValueState>* GetNode(ITERATOR I) {
  return *I;
}

template <> static inline
ExplodedNode<ValueState>* GetNode(GRExprEngine::undef_arg_iterator I) {
  return I->first;
}

//===----------------------------------------------------------------------===//
// Analysis Driver.
//===----------------------------------------------------------------------===//

template <typename ITERATOR>
static void EmitWarning(Diagnostic& Diag, PathDiagnosticClient* PD,
                        ASTContext& Ctx, BugReporter& BR,
                        const BugDescription& Desc,
                        ExplodedGraph<GRExprEngine>& G,
                        ITERATOR I, ITERATOR E) {
  
  for (; I != E; ++I)
    BR.EmitPathWarning(Diag, PD, Ctx, Desc, G, GetNode(I));
}

namespace clang {
  
unsigned RunGRSimpleVals(CFG& cfg, Decl& CD, ASTContext& Ctx,
                         Diagnostic& Diag, PathDiagnosticClient* PD,
                         bool Visualize, bool TrimGraph) {
  
  GRCoreEngine<GRExprEngine> Eng(cfg, CD, Ctx);
  GRExprEngine* CS = &Eng.getCheckerState();
  
  // Set base transfer functions.
  GRSimpleVals GRSV;
  CS->setTransferFunctions(GRSV);
  
  // Add extra checkers.
  llvm::OwningPtr<GRSimpleAPICheck> FoundationCheck(
    CreateBasicObjCFoundationChecks(Ctx, &CS->getStateManager()));
  
  CS->AddObjCMessageExprCheck(FoundationCheck.get());
  
  // Execute the worklist algorithm.
  Eng.ExecuteWorkList(120000);
  
  BugReporter BR;
  ExplodedGraph<GRExprEngine>& G = Eng.getGraph();
  
  EmitWarning(Diag, PD, Ctx, BR, NullDeref(), G,
              CS->null_derefs_begin(), CS->null_derefs_end());


  EmitWarning(Diag, PD, Ctx, BR, UndefDeref(), G,
              CS->undef_derefs_begin(), CS->undef_derefs_end());

  EmitWarning(Diag, PD, Ctx, BR, UndefBranch(), G,
              CS->undef_branches_begin(), CS->undef_branches_end());
  
  EmitWarning(Diag, PD, Ctx, BR, DivZero(), G,
              CS->explicit_bad_divides_begin(), CS->explicit_bad_divides_end());
  
  EmitWarning(Diag, PD, Ctx, BR, UndefResult(), G,
              CS->undef_results_begin(), CS->undef_results_end());
  
  EmitWarning(Diag, PD, Ctx, BR, BadCall(), G,
              CS->bad_calls_begin(), CS->bad_calls_end());
  
  EmitWarning(Diag, PD, Ctx, BR, BadArg(), G,
              CS->undef_arg_begin(), CS->undef_arg_end());
  
  EmitWarning(Diag, PD, Ctx, BR, BadMsgExprArg(), G,
              CS->msg_expr_undef_arg_begin(), CS->msg_expr_undef_arg_end());
  
  for (GRExprEngine::UndefReceiversTy::iterator I = CS->undef_receivers_begin(),
                                  E = CS->undef_receivers_end(); I!=E; ++I) {
       
    BadReceiver Desc(*I);
    BR.EmitPathWarning(Diag, PD, Ctx, Desc, G, *I);
  }
  
  EmitWarning(Diag, PD, Ctx, BR, RetStack(), G,
              CS->ret_stackaddr_begin(), CS->ret_stackaddr_end());

  
  FoundationCheck.get()->ReportResults(Diag, PD, Ctx, BR, G);
#ifndef NDEBUG
  if (Visualize) CS->ViewGraph(TrimGraph);
#endif
  
  return Eng.getGraph().size();
}
  
} // end clang namespace

//===----------------------------------------------------------------------===//
// Transfer function for Casts.
//===----------------------------------------------------------------------===//

RVal GRSimpleVals::EvalCast(GRExprEngine& Eng, NonLVal X, QualType T) {
  
  if (!isa<nonlval::ConcreteInt>(X))
    return UnknownVal();

  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  llvm::APSInt V = cast<nonlval::ConcreteInt>(X).getValue();
  V.setIsUnsigned(T->isUnsignedIntegerType() || T->isPointerType() 
                  || T->isObjCQualifiedIdType());
  V.extOrTrunc(Eng.getContext().getTypeSize(T));
  
  if (T->isPointerType())
    return lval::ConcreteInt(BasicVals.getValue(V));
  else
    return nonlval::ConcreteInt(BasicVals.getValue(V));
}

// Casts.

RVal GRSimpleVals::EvalCast(GRExprEngine& Eng, LVal X, QualType T) {
  
  if (T->isPointerLikeType() || T->isObjCQualifiedIdType())
    return X;
  
  assert (T->isIntegerType());
  
  if (!isa<lval::ConcreteInt>(X))
    return UnknownVal();
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  llvm::APSInt V = cast<lval::ConcreteInt>(X).getValue();
  V.setIsUnsigned(T->isUnsignedIntegerType() || T->isPointerType());
  V.extOrTrunc(Eng.getContext().getTypeSize(T));

  return nonlval::ConcreteInt(BasicVals.getValue(V));
}

// Unary operators.

RVal GRSimpleVals::EvalMinus(GRExprEngine& Eng, UnaryOperator* U, NonLVal X){
  
  switch (X.getSubKind()) {
      
    case nonlval::ConcreteIntKind:
      return cast<nonlval::ConcreteInt>(X).EvalMinus(Eng.getBasicVals(), U);
      
    default:
      return UnknownVal();
  }
}

RVal GRSimpleVals::EvalComplement(GRExprEngine& Eng, NonLVal X) {

  switch (X.getSubKind()) {
      
    case nonlval::ConcreteIntKind:
      return cast<nonlval::ConcreteInt>(X).EvalComplement(Eng.getBasicVals());
      
    default:
      return UnknownVal();
  }
}

// Binary operators.

RVal GRSimpleVals::EvalBinOp(GRExprEngine& Eng, BinaryOperator::Opcode Op,
                             NonLVal L, NonLVal R)  {
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  while (1) {
    
    switch (L.getSubKind()) {
      default:
        return UnknownVal();
        
      case nonlval::ConcreteIntKind:
        
        if (isa<nonlval::ConcreteInt>(R)) {          
          const nonlval::ConcreteInt& L_CI = cast<nonlval::ConcreteInt>(L);
          const nonlval::ConcreteInt& R_CI = cast<nonlval::ConcreteInt>(R);          
          return L_CI.EvalBinOp(BasicVals, Op, R_CI);          
        }
        else {
          NonLVal tmp = R;
          R = L;
          L = tmp;
          continue;
        }
        
      case nonlval::SymbolValKind: {
        
        if (isa<nonlval::ConcreteInt>(R)) {
          const SymIntConstraint& C =
            BasicVals.getConstraint(cast<nonlval::SymbolVal>(L).getSymbol(), Op,
                                    cast<nonlval::ConcreteInt>(R).getValue());
          
          return nonlval::SymIntConstraintVal(C);
        }
        else
          return UnknownVal();
      }
    }
  }
}


// Binary Operators (except assignments and comma).

RVal GRSimpleVals::EvalBinOp(GRExprEngine& Eng, BinaryOperator::Opcode Op,
                             LVal L, LVal R) {
  
  switch (Op) {

    default:
      return UnknownVal();
      
    case BinaryOperator::EQ:
      return EvalEQ(Eng, L, R);
      
    case BinaryOperator::NE:
      return EvalNE(Eng, L, R);      
  }
}

// Pointer arithmetic.

RVal GRSimpleVals::EvalBinOp(GRExprEngine& Eng, BinaryOperator::Opcode Op,
                             LVal L, NonLVal R) {  
  return UnknownVal();
}

// Equality operators for LVals.

RVal GRSimpleVals::EvalEQ(GRExprEngine& Eng, LVal L, LVal R) {
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  switch (L.getSubKind()) {

    default:
      assert(false && "EQ not implemented for this LVal.");
      return UnknownVal();
      
    case lval::ConcreteIntKind:

      if (isa<lval::ConcreteInt>(R)) {
        bool b = cast<lval::ConcreteInt>(L).getValue() ==
                 cast<lval::ConcreteInt>(R).getValue();
        
        return NonLVal::MakeIntTruthVal(BasicVals, b);
      }
      else if (isa<lval::SymbolVal>(R)) {
        
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<lval::SymbolVal>(R).getSymbol(),
                               BinaryOperator::EQ,
                               cast<lval::ConcreteInt>(L).getValue());
        
        return nonlval::SymIntConstraintVal(C);
      }
      
      break;
      
    case lval::SymbolValKind: {

      if (isa<lval::ConcreteInt>(R)) {          
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<lval::SymbolVal>(L).getSymbol(),
                               BinaryOperator::EQ,
                               cast<lval::ConcreteInt>(R).getValue());
        
        return nonlval::SymIntConstraintVal(C);
      }
      
      // FIXME: Implement == for lval Symbols.  This is mainly useful
      //  in iterator loops when traversing a buffer, e.g. while(z != zTerm).
      //  Since this is not useful for many checkers we'll punt on this for 
      //  now.
       
      return UnknownVal();      
    }
      
    case lval::DeclValKind:
    case lval::FuncValKind:
    case lval::GotoLabelKind:
      return NonLVal::MakeIntTruthVal(BasicVals, L == R);
  }
  
  return NonLVal::MakeIntTruthVal(BasicVals, false);
}

RVal GRSimpleVals::EvalNE(GRExprEngine& Eng, LVal L, LVal R) {
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();

  switch (L.getSubKind()) {

    default:
      assert(false && "NE not implemented for this LVal.");
      return UnknownVal();
      
    case lval::ConcreteIntKind:
      
      if (isa<lval::ConcreteInt>(R)) {
        bool b = cast<lval::ConcreteInt>(L).getValue() !=
                 cast<lval::ConcreteInt>(R).getValue();
        
        return NonLVal::MakeIntTruthVal(BasicVals, b);
      }
      else if (isa<lval::SymbolVal>(R)) {        
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<lval::SymbolVal>(R).getSymbol(),
                                  BinaryOperator::NE,
                                  cast<lval::ConcreteInt>(L).getValue());
        
        return nonlval::SymIntConstraintVal(C);
      }
      
      break;
      
    case lval::SymbolValKind: {
      if (isa<lval::ConcreteInt>(R)) {          
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<lval::SymbolVal>(L).getSymbol(),
                                  BinaryOperator::NE,
                                  cast<lval::ConcreteInt>(R).getValue());
        
        return nonlval::SymIntConstraintVal(C);
      }
      
      // FIXME: Implement != for lval Symbols.  This is mainly useful
      //  in iterator loops when traversing a buffer, e.g. while(z != zTerm).
      //  Since this is not useful for many checkers we'll punt on this for 
      //  now.
      
      return UnknownVal();
      
      break;
    }
      
    case lval::DeclValKind:
    case lval::FuncValKind:
    case lval::GotoLabelKind:
      return NonLVal::MakeIntTruthVal(BasicVals, L != R);
  }
  
  return NonLVal::MakeIntTruthVal(BasicVals, true);
}

//===----------------------------------------------------------------------===//
// Transfer function for Function Calls.
//===----------------------------------------------------------------------===//

void GRSimpleVals::EvalCall(ExplodedNodeSet<ValueState>& Dst,
                            GRExprEngine& Eng,
                            GRStmtNodeBuilder<ValueState>& Builder,
                            CallExpr* CE, LVal L,
                            ExplodedNode<ValueState>* Pred) {
  
  ValueStateManager& StateMgr = Eng.getStateManager();
  ValueState* St = Builder.GetState(Pred);
  
  // Invalidate all arguments passed in by reference (LVals).

  for (CallExpr::arg_iterator I = CE->arg_begin(), E = CE->arg_end();
        I != E; ++I) {

    RVal V = StateMgr.GetRVal(St, *I);
    
    if (isa<LVal>(V))
      St = StateMgr.SetRVal(St, cast<LVal>(V), UnknownVal());
  }
  
  // Make up a symbol for the return value of this function.
  
  if (CE->getType() != Eng.getContext().VoidTy) {    
    unsigned Count = Builder.getCurrentBlockCount();
    SymbolID Sym = Eng.getSymbolManager().getConjuredSymbol(CE, Count);
        
    RVal X = CE->getType()->isPointerType() 
             ? cast<RVal>(lval::SymbolVal(Sym)) 
             : cast<RVal>(nonlval::SymbolVal(Sym));
    
    St = StateMgr.SetRVal(St, CE, X, Eng.getCFG().isBlkExpr(CE), false);
  }  
    
  Builder.MakeNode(Dst, CE, Pred, St);
}
